"""Model loading, planning and closed-loop rollout for Push-T (World Model + DiT).

Three policies share the same frozen World Model encoder (frame -> s_t):

- "dit_id": DiT draws the object path, the inverse dynamics turns it into actions (course sketch).
- "dit_wm": same path, but the actions are chosen with the World Model dynamics: candidate action
  sequences around the inverse-dynamics proposal are imagined with the dynamics model and the one
  whose predicted object trajectory best matches the DiT path (FDE) is executed.
- "action_diffusion": Diffusion-Policy-style baseline that diffuses the actions directly.

Evaluation protocol used everywhere: coverage from gym-pusht (intersection / goal area),
success = coverage > 0.95, 300 steps. Heuristics outside the learned policy are opt-in flags.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import normalize as _l2normalize

from pusht_wm_dit.envs.pusht_sim import PushTSim, VideoRecorder
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.dynamics import PushDynamics, load_dynamics
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.models.world_model import WorldModel
from pusht_wm_dit.utils.metrics import compute_fde_numpy
from pusht_wm_dit.utils.normalizer import normalize_image, normalize_pos, unnormalize_pos

GOAL_XY = (256.0, 256.0)
GOAL_THETA = math.pi / 4.0
SUCCESS_THRESHOLD = 0.95  # gym-pusht success_threshold
MAX_STEP_PX = 40.0        # per-step pusher displacement limit (rate limiting)
POLICY_KINDS = ("dit_id", "dit_wm", "action_diffusion", "dit_id_vision", "dit_vwm")
# dit_id_vision / dit_vwm use no simulator block state at run time: the block pose comes from the
# image (pose probe on the vision World Model's features: self-trained JEPA encoder, or legacy DINOv2)
# and dit_vwm chooses actions with the vision World Model.


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_ckpt(path: str, ckpt_dir: Optional[str]) -> str:
    resolved = os.path.join(ckpt_dir, os.path.basename(path)) if ckpt_dir else path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _load_ckpt(path: str, device: str) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_world_model(config: Dict[str, Any], device: str, ckpt_dir: Optional[str] = None) -> WorldModel:
    w = config["world_model"]
    wm = WorldModel(latent_dim=w["latent_dim"], agent_pos_dim=w["agent_pos_dim"], pose_dim=w["pose_dim"],
                    action_dim=w["action_dim"], num_groups=w["num_groups"],
                    transition_hidden_dim=w["transition_hidden_dim"]).to(device)
    wm.load_state_dict(_load_ckpt(_resolve_ckpt(w["save_path"], ckpt_dir), device)["model_state_dict"])
    wm.freeze()
    return wm


def _make_dit(config: Dict[str, Any], latent_dim: int, traj_dim: int, horizon: int, device: str) -> DiTTrajectoryPlanner:
    d = config["dit_planner"]
    return DiTTrajectoryPlanner(horizon=horizon, traj_dim=traj_dim, latent_dim=latent_dim, model_dim=d["model_dim"],
                                num_heads=d["num_heads"], num_layers=d["num_layers"], mlp_ratio=d["mlp_ratio"],
                                diffusion_steps=d["diffusion_steps"], beta_schedule=d["beta_schedule"],
                                w_xy=d["w_xy"], w_theta=d["w_theta"], lambda_fde=d["lambda_fde"]).to(device)


def load_models(
    config: Dict[str, Any],
    device: str,
    ckpt_dir: Optional[str] = None,
) -> Tuple[WorldModel, DiTTrajectoryPlanner, InverseDynamicsController]:
    """World Model, DiT planner and inverse dynamics. The observation history (1 or 2 frames)
    is read from the planner checkpoint, so old checkpoints keep working."""
    wm = load_world_model(config, device, ckpt_dir)
    lat = config["world_model"]["latent_dim"]
    dit_ck = _load_ckpt(_resolve_ckpt(config["dit_planner"]["save_path"], ckpt_dir), device)
    hist = int(dit_ck.get("obs_history", 1))
    dit = _make_dit(config, lat * hist, config["dit_planner"]["traj_dim"], config["dit_planner"]["horizon"], device)
    dit.load_state_dict(dit_ck["model_state_dict"])
    dit.eval()
    dit.obs_history = hist

    i = config["inverse_dynamics"]
    id_ck = _load_ckpt(_resolve_ckpt(i["save_path"], ckpt_dir), device)
    id_ctrl = InverseDynamicsController(latent_dim=lat * int(id_ck.get("obs_history", 1)), horizon=config["dit_planner"]["horizon"],
                                        traj_dim=config["dit_planner"]["traj_dim"], action_dim=i["action_dim"],
                                        exec_steps=i["exec_steps"], hidden_dims=tuple(i["hidden_dims"])).to(device)
    id_ctrl.load_state_dict(id_ck["model_state_dict"])
    id_ctrl.eval()
    return wm, dit, id_ctrl


def load_dynamics_model(config: Dict[str, Any], device: str, ckpt_dir: Optional[str] = None) -> Optional[PushDynamics]:
    path = config.get("dynamics", {}).get("save_path")
    if not path:
        return None
    path = os.path.join(ckpt_dir, os.path.basename(path)) if ckpt_dir else path
    return load_dynamics(path, device) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class _EncoderMixin:
    @torch.no_grad()
    def encode(self, obs: Dict[str, Any], history: int) -> torch.Tensor:
        def enc(pixels, agent):
            f = torch.from_numpy(normalize_image(pixels)).float().unsqueeze(0).to(self.device)
            a = torch.from_numpy(np.asarray(agent, dtype=np.float32)).unsqueeze(0).to(self.device)
            return self.wm.encode(f, a)
        s = enc(obs["pixels"], obs["agent_pos"])
        if history == 2:
            s = torch.cat([s, enc(obs.get("prev_pixels", obs["pixels"]), obs.get("prev_agent_pos", obs["agent_pos"]))], -1)
        return s

    @torch.no_grad()
    def imagine(self, obs: Dict[str, Any], actions: np.ndarray) -> Optional[np.ndarray]:
        """World Model imagination: block poses (K, 3) in pixels/radians after each action, or None."""
        if self.dynamics is None:
            return None
        b, a, v = _state_tensors(obs, self.device)
        acts = torch.from_numpy(np.asarray(actions, dtype=np.float32)).unsqueeze(0).to(self.device)
        blocks = self.dynamics.rollout(b, a, v, acts)[0][0].cpu().numpy()
        return np.stack([blocks[:, 0] * 256 + 256, blocks[:, 1] * 256 + 256, blocks[:, 2]], -1)


def estimated_coverage(pose4: np.ndarray) -> float:
    """Coverage of the goal (intersection / goal area) for a block pose [x, y, sin, cos] in normalised units."""
    from pusht_wm_dit.utils.metrics import create_tee_polygon
    x, y = pose4[0] * 256 + 256, pose4[1] * 256 + 256
    th = math.atan2(pose4[2], pose4[3])
    goal = create_tee_polygon(GOAL_XY[0], GOAL_XY[1], GOAL_THETA)
    return float(goal.intersection(create_tee_polygon(x, y, th)).area / goal.area)


def _state_tensors(obs: Dict[str, Any], device: str, n: int = 1):
    bp = obs["block_pose"]
    block = np.array([[(bp[0] - 256.0) / 256.0, (bp[1] - 256.0) / 256.0, float(bp[2])]], dtype=np.float32)
    agent = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(1, 2)
    vel = np.asarray(obs.get("agent_vel", np.zeros(2)), dtype=np.float32).reshape(1, 2)
    return tuple(torch.from_numpy(a).to(device).expand(n, -1) for a in (block, agent, vel))


class WMDiTPolicy(_EncoderMixin):
    """World Model encoder -> DiT object path -> actions (inverse dynamics, optionally chosen by the WM)."""

    def __init__(self, wm, dit, id_ctrl, device, sample_steps: int = 10, best_of_n: int = 8,
                 dynamics: Optional[PushDynamics] = None, action_mode: str = "id",
                 num_candidates: int = 64, vision: Optional[Dict[str, Any]] = None, pose_source: str = "true"):
        self.wm, self.dit, self.id_ctrl, self.device = wm, dit, id_ctrl, device
        self.sample_steps, self.best_of_n = sample_steps, best_of_n
        self.dynamics = dynamics
        self.vision = vision  # {"encoder", "norm", "predictor", "probe"} or None
        self.action_mode = action_mode if (dynamics is not None or action_mode != "wm") else "id"
        self.pose_source = pose_source if vision is not None else "true"
        self.num_candidates = num_candidates
        self.obs_history = getattr(dit, "obs_history", 1)
        # dit_vwm: hand over to the inverse dynamics once the image-estimated coverage exceeds this value
        # (the vision World Model is ~4-6 px accurate, too coarse for the last few pixels of docking)
        self.vwm_gate_coverage: Optional[float] = None
        self.frames: List[Tuple[np.ndarray, np.ndarray]] = []
        self.last: Dict[str, Any] = {}

    @property
    def kind(self):
        if self.action_mode == "vwm":
            return "dit_vwm"
        if self.pose_source == "vision":
            return "dit_id_vision"
        return "dit_wm" if self.action_mode == "wm" else "dit_id"

    # frame buffer for the vision World Model (needs the frame `skip` steps ago)
    def reset(self, obs: Dict[str, Any]):
        self.frames = [(obs["pixels"], np.asarray(obs["agent_pos"], np.float32))]

    def observe(self, obs: Dict[str, Any]):
        self.frames.append((obs["pixels"], np.asarray(obs["agent_pos"], np.float32)))
        skip = self.vision["predictor"].skip if self.vision else 1
        self.frames = self.frames[-(skip + 1):]

    @torch.no_grad()
    def _vision_state(self, obs):
        """Normalised DINOv2 features of the current and the `skip`-steps-old frame, and their pusher positions."""
        if not self.frames:
            self.reset(obs)
        old_px, old_ag = self.frames[0]
        z = self.vision["encoder"].encode(np.stack([old_px, obs["pixels"]]))
        zn = self.vision["norm"](z.to(self.device))
        ags = torch.from_numpy(np.stack([old_ag, np.asarray(obs["agent_pos"], np.float32)])).to(self.device)
        return zn[0:1], zn[1:2], ags[0:1], ags[1:2]

    @torch.no_grad()
    def plan(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (actions (Ta, 2) in [-1, 1], planned block path (H, 4) in [-1, 1])."""
        s_t = self.encode(obs, self.obs_history)
        agent = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0).to(self.device)
        vis = self._vision_state(obs) if (self.pose_source == "vision" or self.action_mode == "vwm") else None
        if self.pose_source == "vision":
            p = self.vision["probe"](vis[1])
            pose = torch.cat([p[:, :2].clamp(-1, 1), _l2normalize(p[:, 2:], dim=-1)], -1)  # pose read from the image
        else:
            pose = torch.from_numpy(np.asarray(obs["block_pose_4d"], dtype=np.float32)).unsqueeze(0).to(self.device)
        path = self.dit.sample(s_t, num_inference_steps=self.sample_steps, current_pose_4d=pose, best_of_n=self.best_of_n)
        s_id = s_t if self.id_ctrl.latent_dim == s_t.shape[-1] else s_t[:, : self.id_ctrl.latent_dim]
        actions = self.id_ctrl(s_id, path, agent_pos=agent)[0]
        if self.action_mode == "wm":
            actions = self._choose_with_world_model(obs, actions, path[0])
        elif self.action_mode == "vwm":
            est = estimated_coverage(pose[0].cpu().numpy())
            self.last_gate = est
            if self.vwm_gate_coverage is None or est < self.vwm_gate_coverage:
                actions = self._choose_with_vision_world_model(vis, actions, path[0])
        actions_np, path_np = actions.cpu().numpy(), path[0].cpu().numpy()
        self.last = {"path": path_np, "actions": actions_np, "pose_used": pose[0].cpu().numpy()}
        return actions_np, path_np

    @torch.no_grad()
    def imagine(self, obs: Dict[str, Any], actions: np.ndarray) -> Optional[np.ndarray]:
        """Imagined block poses (pixels / radians). Vision modes use the vision World Model (one pose per
        predictor step), the others the state dynamics model (one pose per action)."""
        if self.vision is not None and (self.action_mode == "vwm" or self.pose_source == "vision"):
            from pusht_wm_dit.models.latent_world_model import pose4_to_xyth
            pred = self.vision["predictor"]
            acts = np.asarray(actions, dtype=np.float32)
            k = len(acts) // pred.skip
            if k == 0:
                return None
            z_old, z_cur, a_old, a_cur = self._vision_state(obs)
            zs, _ = pred.rollout(z_old, z_cur, a_old, a_cur, torch.from_numpy(acts[: k * pred.skip]).unsqueeze(0).to(self.device))
            p = pose4_to_xyth(self.vision["probe"](zs[0])).cpu().numpy()
            return np.stack([p[:, 0] * 256 + 256, p[:, 1] * 256 + 256, p[:, 2]], -1)
        return super().imagine(obs, actions)

    def _candidates(self, proposal: torch.Tensor) -> torch.Tensor:
        n, ta = self.num_candidates, proposal.shape[0]
        g = torch.randn(n, ta, 2, device=self.device)
        for t in range(1, ta):  # temporally smooth perturbations
            g[:, t] = 0.7 * g[:, t - 1] + 0.71 * g[:, t]
        scales = torch.tensor([0.0, 0.02, 0.04, 0.08], device=self.device).repeat_interleave(n // 4)[:, None, None]
        cand = torch.clamp(proposal[None] + g * scales, -1, 1)
        cand[0] = proposal
        return cand

    @staticmethod
    def _path_cost(blocks_xyth: torch.Tensor, target4: torch.Tensor) -> torch.Tensor:
        """blocks (n, k, 3) normalised xy + angle vs target path points (k, 4) -> (n,) FDE-style cost in px/deg."""
        pos = torch.linalg.norm(blocks_xyth[..., :2] - target4[None, :, :2], dim=-1) * 256.0
        d = blocks_xyth[..., 2] - torch.atan2(target4[:, 2], target4[:, 3])[None]
        ang = torch.rad2deg(torch.abs(torch.atan2(torch.sin(d), torch.cos(d))))
        return (pos ** 2 + ang ** 2).mean(-1)

    def _choose_with_vision_world_model(self, vis, proposal: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Imagine candidates with the vision World Model (image features only) and keep the one whose
        imagined block motion, read by the pose probe, best follows the DiT path."""
        from pusht_wm_dit.models.latent_world_model import pose4_to_xyth
        pred, probe = self.vision["predictor"], self.vision["probe"]
        cand = self._candidates(proposal)
        n, ta = cand.shape[:2]
        z_old, z_cur, a_old, a_cur = vis
        zs, _ = pred.rollout(z_old.expand(n, -1, -1), z_cur.expand(n, -1, -1), a_old.expand(n, -1), a_cur.expand(n, -1), cand)
        k = zs.shape[1]
        poses = pose4_to_xyth(probe(zs.reshape(n * k, *zs.shape[2:]))).view(n, k, 3)
        target = path[[pred.skip * (j + 1) for j in range(k)]]          # path[0] = current pose
        cost = self._path_cost(poses, target)
        best = int(torch.argmin(cost))
        self.last_choice = {"cost_proposal": float(cost[0]), "cost_best": float(cost[best]), "index": best}
        self.last_imagined = torch.stack([poses[best, :, 0] * 256 + 256, poses[best, :, 1] * 256 + 256, poses[best, :, 2]], -1).cpu().numpy()
        return cand[best]

    def _choose_with_world_model(self, obs, proposal: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Imagine candidate action sequences and keep the one whose predicted object motion best follows the DiT path."""
        cand = self._candidates(proposal)
        n, ta = cand.shape[:2]
        b, a, v = _state_tensors(obs, self.device, n)
        blocks = self.dynamics.rollout(b, a, v, cand)[0]                       # (n, ta, 3)
        cost = self._path_cost(blocks, path[1: ta + 1])                        # path[0] is the current pose
        best = int(torch.argmin(cost))
        self.last_choice = {"cost_proposal": float(cost[0]), "cost_best": float(cost[best]), "index": best}
        return cand[best]


class ActionDiffusionPolicy(_EncoderMixin):
    """Diffusion-Policy-style baseline: the DiT generates 16 actions, the first Ta are executed."""

    kind = "action_diffusion"

    def __init__(self, wm, model, device, sample_steps: int = 10, exec_steps: int = 8, dynamics=None):
        self.wm, self.model, self.device = wm, model, device
        self.sample_steps, self.exec_steps = sample_steps, exec_steps
        self.obs_history = getattr(model, "obs_history", 1)
        self.dynamics = dynamics
        self.last: Dict[str, Any] = {}

    @torch.no_grad()
    def plan(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        acts = self.model.sample(self.encode(obs, self.obs_history), num_inference_steps=self.sample_steps, best_of_n=1)
        a = acts[0, : self.exec_steps].cpu().numpy()
        self.last = {"path": None, "actions": a}
        return a, None


def build_policy(config: Dict[str, Any], device: str, ckpt_dir: Optional[str] = None, kind: str = "dit_id",
                 vision_backend: Optional[str] = None):
    """kind: one of POLICY_KINDS. vision_backend (vision kinds only): 'jepa_wm' | 'vjepa2_wm' | 'dinov2_hf_wm'
    (default: config['vision_backend'])."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown policy kind {kind!r}; choose from {POLICY_KINDS}")
    dynamics = load_dynamics_model(config, device, ckpt_dir)
    if kind == "action_diffusion":
        wm = load_world_model(config, device, ckpt_dir)
        acfg = config["action_diffusion"]
        ck = _load_ckpt(_resolve_ckpt(acfg["save_path"], ckpt_dir), device)
        hist = int(ck.get("obs_history", 1))
        model = _make_dit(config, config["world_model"]["latent_dim"] * hist, 2, acfg["horizon"], device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        model.obs_history = hist
        return ActionDiffusionPolicy(wm, model, device, config["dit_planner"]["sample_steps"],
                                     config["evaluation"]["exec_horizon"], dynamics=dynamics)
    if kind == "dit_wm" and dynamics is None:
        raise FileNotFoundError("dit_wm needs the dynamics model: run scripts/3_world_model/train_dynamics.py first")
    vision = None
    if kind in ("dit_id_vision", "dit_vwm"):
        vision = load_vision_bundle(config, device, ckpt_dir, backend=vision_backend)
        if vision is None:
            raise FileNotFoundError("vision modes need a vision World Model: run scripts/6_vision_world_model/train_jepa_wm.py "
                                    "(self-trained JEPA) or scripts/6_vision_world_model/train_frozen_encoder_wm.py (frozen V-JEPA 2 / DINOv2) first")
    wm, dit, id_ctrl = load_models(config, device, ckpt_dir=ckpt_dir)
    mode = {"dit_wm": "wm", "dit_vwm": "vwm"}.get(kind, "id")
    pol = WMDiTPolicy(wm, dit, id_ctrl, device, sample_steps=config["dit_planner"]["sample_steps"],
                      dynamics=dynamics, action_mode=mode, vision=vision,
                      pose_source="vision" if kind in ("dit_id_vision", "dit_vwm") else "true")
    pol.vwm_gate_coverage = config.get("vision_wm", {}).get("gate_coverage")
    return pol


def load_vision_bundle(config: Dict[str, Any], device: str, ckpt_dir: Optional[str] = None,
                       backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Vision World Model (encoder, predictor, feature norm, pose probe), or None if not trained.
    backend (or config['vision_backend']): 'jepa_wm' = self-trained JEPA encoder (train_jepa_wm.py), 'vjepa2_wm' /
    'dinov2_hf_wm' = frozen pretrained encoder + predictor trained on lerobot/pusht (train_frozen_encoder_wm.py),
    'vision_wm' = legacy frozen DINOv2 version of stage E5a (experiments/E5_vision_world_models/code_legacy_dinov2).
    Without a backend: the self-trained JEPA, else the legacy DINOv2 version."""
    from pusht_wm_dit.models.latent_world_model import load_vision_wm
    backend = backend or config.get("vision_backend")
    for key in ((backend,) if backend else ("jepa_wm", "vision_wm")):
        path = config.get(key, {}).get("save_path")
        if not path:
            continue
        alt = os.path.join(ckpt_dir, os.path.basename(path)) if ckpt_dir else None
        path = alt if alt and os.path.exists(alt) else path
        if not os.path.exists(path):
            continue
        bundle = load_vision_wm(path, device)
        if "encoder" in bundle:
            from pusht_wm_dit.models.jepa_encoder import FrameEncoder
            bundle["encoder"] = FrameEncoder(bundle["encoder"], device)
        elif bundle.get("encoder_type"):
            from pusht_wm_dit.models.frozen_encoders import get_frozen_encoder
            bundle["encoder"] = get_frozen_encoder(bundle["encoder_type"], device)
        else:
            from pusht_wm_dit.models.dinov2_encoder import get_encoder
            bundle["encoder"] = get_encoder(device)
        bundle["source"] = key
        return bundle
    return None


# ---------------------------------------------------------------------------
# Action post-processing & optional heuristics
# ---------------------------------------------------------------------------
_shared_guard = None


def get_shared_guard():
    """NonPenetrationGuard of the archived hand-written controllers (experiments/E2_handcrafted_controllers/code).
    Only used when use_collision_guard=True (stage E2); the final system never calls it."""
    global _shared_guard
    if _shared_guard is None:
        import sys
        from pusht_wm_dit.paths import ROOT
        code_dir = os.path.join(ROOT, "experiments", "E2_handcrafted_controllers", "code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from controllers.collision_guard import NonPenetrationGuard
        _shared_guard = NonPenetrationGuard()
    return _shared_guard


def postprocess_action(
    act_norm: np.ndarray,
    obs: Dict[str, Any],
    use_stance_heuristic: bool = False,
    max_step_px: float = MAX_STEP_PX,
    block_body: Optional[Any] = None,
    use_collision_guard: bool = False,
) -> np.ndarray:
    """Per-step rate limiting, plus the optional (archived) stance heuristic and collision guard."""
    act_px = unnormalize_pos(np.asarray(act_norm, dtype=np.float32))
    agent_px = np.asarray(obs["agent_pos_px"], dtype=np.float32)

    if use_stance_heuristic:
        bx, by = float(obs["block_pose"][0]), float(obs["block_pose"][1])
        ax, ay = float(agent_px[0]), float(agent_px[1])
        gx, gy = GOAL_XY
        dist_g = math.hypot(gx - bx, gy - by)
        if dist_g > 35.0:
            ux, uy = (gx - bx) / (dist_g + 1e-6), (gy - by) / (dist_g + 1e-6)
            rx, ry = ax - bx, ay - by
            if rx * ux + ry * uy > 20.0:
                perp_x, perp_y = -uy, ux
                side = 1.0 if (rx * perp_x + ry * perp_y) >= 0 else -1.0
                act_px = np.array([bx + side * perp_x * 75.0 - ux * 25.0, by + side * perp_y * 75.0 - uy * 25.0], dtype=np.float32)

    if use_collision_guard and block_body is not None:
        act_px, _, _ = get_shared_guard().filter_action(agent_px, act_px, block_body)

    delta = act_px - agent_px
    dist = float(np.linalg.norm(delta))
    if dist > max_step_px:
        act_px = agent_px + delta / dist * max_step_px
    return np.asarray(normalize_pos(act_px), dtype=np.float32)


def should_lock(cov: float, max_cov: float, lock_cov: float = 0.82, peak_cov: float = 0.72, drop: float = 0.04) -> bool:
    """Target-lock heuristic (archived): stop when coverage is high or falling after a good peak."""
    return cov >= lock_cov or (max_cov >= peak_cov and cov < max_cov - drop)


def angle_error_to_goal(theta: float) -> float:
    return abs((theta - GOAL_THETA + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------
def run_episode(
    sim: PushTSim,
    policy,
    seed: Optional[int] = None,
    init_state: Optional[Sequence[float]] = None,
    max_steps: int = 300,
    exec_horizon: int = 8,
    use_stance_heuristic: bool = False,
    early_stop: bool = False,
    recorder: Optional[VideoRecorder] = None,
) -> Dict[str, Any]:
    """One receding-horizon rollout. init_state = [agent_x, agent_y, block_x, block_y, block_theta]."""
    if init_state is not None:
        sim.reset(seed=seed)
        obs = sim.set_state(init_state[:2], init_state[2:5])
    else:
        obs, _ = sim.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(obs)

    init_obs = obs
    max_cov = obs["coverage"]
    cov_history: List[float] = [obs["coverage"]]
    fde_history: List[float] = []
    step, end_reason = 0, "MAX_STEPS"
    done = obs["coverage"] > SUCCESS_THRESHOLD
    if done:
        end_reason = "SUCCESS"
    if recorder is not None:
        recorder.record_frame(frame=obs["pixels"], step=0, coverage=obs["coverage"], info_text="Policy initialized")

    while not done and step < max_steps:
        actions, path = policy.plan(obs)
        for a_idx in range(min(exec_horizon, len(actions))):
            act = postprocess_action(actions[a_idx], obs, use_stance_heuristic=use_stance_heuristic)
            obs, _, terminated, truncated, _ = sim.step(act, normalized_action=True)
            if hasattr(policy, "observe"):
                policy.observe(obs)
            step += 1
            cov = obs["coverage"]
            cov_history.append(cov)
            max_cov = max(max_cov, cov)
            step_fde = None
            if path is not None:
                step_fde = compute_fde_numpy(obs["block_pose_4d"], path[min(a_idx + 1, len(path) - 1)])
                fde_history.append(step_fde)
            if terminated or cov > SUCCESS_THRESHOLD:
                done, end_reason = True, "SUCCESS"
            elif early_stop and should_lock(cov, max_cov):
                done, end_reason = True, "TARGET_LOCK"
            elif truncated or step >= max_steps:
                done, end_reason = True, "MAX_STEPS"
            if recorder is not None:
                status = end_reason if done else f"RHC substep {a_idx + 1}/{exec_horizon}"
                recorder.record_frame(frame=obs["pixels"], step=step, coverage=cov, fde=step_fde, info_text=status)
            if done:
                break

    def goal_dist(o):
        return float(math.hypot(o["block_pose"][0] - GOAL_XY[0], o["block_pose"][1] - GOAL_XY[1]))

    return {
        "seed": seed,
        "steps": step,
        "end_reason": end_reason,
        "init_block_pose": [float(v) for v in init_obs["block_pose"]],
        "init_agent_pos": [float(v) for v in init_obs["agent_pos_px"]],
        "final_block_pose": [float(v) for v in obs["block_pose"]],
        "init_coverage": float(init_obs["coverage"]),
        "final_coverage": float(obs["coverage"]),
        "max_coverage": float(max_cov),
        "success": bool(max_cov > SUCCESS_THRESHOLD),
        "init_goal_dist": goal_dist(init_obs),
        "final_goal_dist": goal_dist(obs),
        "init_angle_err_deg": math.degrees(angle_error_to_goal(float(init_obs["block_pose"][2]))),
        "final_angle_err_deg": math.degrees(angle_error_to_goal(float(obs["block_pose"][2]))),
        "mean_fde": float(np.mean(fde_history)) if fde_history else 0.0,
        "max_fde": float(np.max(fde_history)) if fde_history else 0.0,
        "coverage_history": [float(c) for c in cov_history],
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates episode results into the standard Push-T metrics."""
    max_cov = np.array([r["max_coverage"] for r in results])
    succ = [r for r in results if r["success"]]
    return {
        "num_episodes": len(results),
        "mean_max_coverage": float(max_cov.mean()),
        "std_max_coverage": float(max_cov.std()),
        "mean_final_coverage": float(np.mean([r["final_coverage"] for r in results])),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "rate_max_cov_ge_0.80": float(np.mean(max_cov >= 0.80)),
        "rate_max_cov_lt_0.50": float(np.mean(max_cov < 0.50)),
        "median_steps_to_success": float(np.median([r["steps"] for r in succ])) if succ else float("nan"),
        "mean_fde": float(np.mean([r["mean_fde"] for r in results])),
        "mean_final_goal_dist": float(np.mean([r["final_goal_dist"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
    }


def resolve_device(config: Dict[str, Any]) -> str:
    return config.get("device", "cuda") if torch.cuda.is_available() else "cpu"

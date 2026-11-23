"""World Model Lookahead Optimal Stopping & Safe Retreat Optimizer.

Enhanced with:
1. Gating Threshold (Ngưỡng chặn Gating): Strictly prevents premature locking or
   freezing when the T-block is in the intermediate maneuvering/rotation phase
   (cov < 0.80 or dist > 18px).
2. Composite Potential Function (Hàm thế năng đa mục tiêu):
   Phi(x, y, theta) = Coverage + w_dist * (1 - dist/200) + w_angle * (1 - dth/pi)
   Allows the controller to distinguish between:
   - "Strategic temporary dips" (Coverage drops slightly while rotating or realigning,
     but angle & distance improve -> Phi improves -> Continue executing!)
   - "True overshoot degradation" (Block is in the goal slot and pushing drops both
     Coverage & Phi -> Trigger LOCK and Safe Retreat!).
3. Safe Retreat Decoupling: Smoothly pulls the pusher 16px away along the contact normal
   to decouple physical contact once peak docking is reached.
"""

from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
import torch
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely import affinity

from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    four_d_to_pose,
)


class PushTCoverageEvaluator:
    """Ultra-fast geometric coverage & composite potential evaluator using Shapely."""

    def __init__(
        self,
        goal_x: float = 256.0,
        goal_y: float = 256.0,
        goal_theta: float = math.pi / 4.0,
        w_dist: float = 0.25,
        w_angle: float = 0.20,
    ):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_theta = goal_theta
        self.w_dist = w_dist
        self.w_angle = w_angle

        # Base T-shape at origin
        p1 = sg.Polygon([(-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)])
        p2 = sg.Polygon([(-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)])
        self.base_t = unary_union([p1, p2])

        deg = math.degrees(goal_theta)
        rot = affinity.rotate(self.base_t, deg, origin=(0.0, 0.0))
        self.goal_geom = affinity.translate(rot, xoff=goal_x, yoff=goal_y)
        self.goal_area = float(self.goal_geom.area)

    def set_goal(self, goal_x: float, goal_y: float, goal_theta: float) -> None:
        """Dynamically reconfigures the target goal pose for real-time tracking."""
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_theta = float(goal_theta)
        deg = math.degrees(self.goal_theta)
        rot = affinity.rotate(self.base_t, deg, origin=(0.0, 0.0))
        self.goal_geom = affinity.translate(rot, xoff=self.goal_x, yoff=self.goal_y)
        self.goal_area = float(self.goal_geom.area)

    def get_geom(self, x: float, y: float, theta: float) -> sg.Polygon:
        deg = math.degrees(theta)
        rot = affinity.rotate(self.base_t, deg, origin=(0.0, 0.0))
        return affinity.translate(rot, xoff=x, yoff=y)

    def compute_coverage(self, x: float, y: float, theta: float) -> float:
        geom = self.get_geom(x, y, theta)
        inter = self.goal_geom.intersection(geom).area
        return float(inter / self.goal_area)

    def compute_composite_potential(
        self, x: float, y: float, theta: float
    ) -> Tuple[float, float, float, float]:
        """Calculates multi-objective alignment score Phi(x, y, theta).

        Returns:
            Tuple of (phi, cov, dist, dth):
                - phi: Composite potential in [0, 1 + w_dist + w_angle]
                - cov: Raw geometric coverage in [0, 1]
                - dist: Euclidean distance to goal in pixels
                - dth: Absolute wrapped angular error in radians
        """
        cov = self.compute_coverage(x, y, theta)
        dist = math.hypot(x - self.goal_x, y - self.goal_y)
        dth = abs((theta - self.goal_theta + math.pi) % (2.0 * math.pi) - math.pi)

        dist_norm = min(1.0, dist / 200.0)
        angle_norm = min(1.0, dth / math.pi)

        phi = cov + self.w_dist * (1.0 - dist_norm) + self.w_angle * (1.0 - angle_norm)
        return float(phi), float(cov), float(dist), float(dth)


def compute_safe_retreat(
    agent_px: np.ndarray,
    block_pos_px: np.ndarray,
    retreat_dist: float = 16.0,
) -> np.ndarray:
    """Pulls the pusher away from the block along the radial contact normal.

    Breaks contact physics cleanly to prevent jitter/impulses from nudging the block.
    """
    vec = np.asarray(agent_px, dtype=np.float32) - np.asarray(block_pos_px, dtype=np.float32)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-4:
        return agent_px + np.array([0.0, retreat_dist], dtype=np.float32)
    u = vec / dist
    return agent_px + u * retreat_dist


class WorldModelLookaheadOptimizer:
    """Predicts optimal stopping horizons and prevents docking overshoot using World Model rollouts."""

    def __init__(
        self,
        gating_cov: float = 0.80,
        lock_threshold: float = 0.88,
        docking_drop_tolerance: float = 0.012,
        w_dist: float = 0.25,
        w_angle: float = 0.20,
        goal_x: float = 256.0,
        goal_y: float = 256.0,
        goal_theta: float = math.pi / 4.0,
    ):
        self.evaluator = PushTCoverageEvaluator(
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
            w_dist=w_dist,
            w_angle=w_angle,
        )
        self.gating_cov = gating_cov
        self.lock_threshold = lock_threshold
        self.docking_drop_tolerance = docking_drop_tolerance

        # State telemetry
        self.last_rollout_covs: List[float] = []
        self.last_rollout_phis: List[float] = []
        self.last_best_k: int = 0
        self.last_decision: str = "IDLE"
        self.last_reason: str = ""
        self.last_peak_cov: float = 0.0
        self.last_peak_phi: float = 0.0

    def set_goal(self, goal_x: float, goal_y: float, goal_theta: float) -> None:
        """Updates target goal pose across the evaluator."""
        self.evaluator.set_goal(goal_x, goal_y, goal_theta)

    @torch.no_grad()
    def evaluate_rollout(
        self,
        wm: Any,
        obs: Dict[str, Any],
        actions: np.ndarray,
        device: str,
    ) -> Dict[str, Any]:
        """Rolls out proposed actions in the World Model latent space with gating and composite potential.

        Args:
            wm: Trained WorldModel module.
            obs: Current observation dict (contains pixels, agent_pos, coverage, block_pose).
            actions: Proposed action array of shape (Ta, 2) in normalized space [-1, 1].
            device: 'cuda' or 'cpu'.

        Returns:
            Dict containing:
                - decision: 'EXECUTE', 'TRUNCATE', or 'LOCK'
                - optimal_k: Best step index (1-based, 1 to Ta)
                - peak_cov: Maximum predicted coverage
                - peak_phi: Maximum predicted composite potential
                - pred_covs: List of predicted coverages at each horizon step
                - pred_phis: List of predicted composite potentials
                - actions_to_execute: Filtered action array
        """
        bx, by = float(obs["block_pose"][0]), float(obs["block_pose"][1])
        bth = float(obs["block_pose"][2])
        phi_now, cov_now, dist_now, dth_now = self.evaluator.compute_composite_potential(bx, by, bth)
        ta = actions.shape[0]

        # 1. Gating Check: Is the block inside the Terminal Docking Basin?
        is_terminal_basin = (cov_now >= self.gating_cov) or (
            cov_now >= 0.72 and dist_now <= 18.0 and dth_now <= 0.20
        )

        # 2. Encode current latent state
        frame = torch.from_numpy(normalize_image(obs["pixels"])).float().unsqueeze(0).to(device)
        agent = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0).to(device)
        s_cur = wm.encode(frame, agent)
        acts_t = torch.from_numpy(np.asarray(actions, dtype=np.float32)).unsqueeze(0).to(device)

        pred_covs: List[float] = []
        pred_phis: List[float] = []
        pred_poses: List[Tuple[float, float, float]] = []

        # 3. Rollout forward latent dynamics
        for k in range(ta):
            act_k = acts_t[:, k]
            s_cur = wm.predict_next_state(s_cur, act_k)
            p_4d, _ = wm.decode(s_cur)
            p_np = p_4d[0].detach().cpu().numpy()
            x, y, th = four_d_to_pose(p_np)
            phi_k, cov_k, _, _ = self.evaluator.compute_composite_potential(float(x), float(y), float(th))
            pred_covs.append(cov_k)
            pred_phis.append(phi_k)
            pred_poses.append((float(x), float(y), float(th)))

        self.last_rollout_covs = pred_covs
        self.last_rollout_phis = pred_phis

        # Find best horizon index according to Composite Potential Phi
        best_k = int(np.argmax(pred_phis))  # 0-based index
        peak_phi = pred_phis[best_k]
        peak_cov = pred_covs[best_k]
        self.last_best_k = best_k + 1
        self.last_peak_cov = peak_cov
        self.last_peak_phi = peak_phi

        # -------------------------------------------------------------
        # 4. DECISION LOGIC: Gating Threshold + Composite Potential
        # -------------------------------------------------------------

        # CASE 1: Outside Terminal Basin (Maneuvering / Aligning Phase)
        # Even if coverage or potential dips temporarily, this is intermediate maneuvering
        # (e.g. pivoting the stem to rotate). DO NOT LOCK or stall!
        if not is_terminal_basin:
            self.last_decision = "EXECUTE"
            self.last_reason = f"Maneuvering (Cov: {cov_now*100:.1f}%, Dist: {dist_now:.1f}px). Gating prevents local trapping."
            return {
                "decision": "EXECUTE",
                "optimal_k": ta,
                "peak_cov": peak_cov,
                "peak_phi": peak_phi,
                "pred_covs": pred_covs,
                "pred_phis": pred_phis,
                "pred_poses": pred_poses,
                "actions_to_execute": actions,
                "reason": self.last_reason,
            }

        # CASE 2: Inside Terminal Basin (Final Docking Phase)

        # 2A. Lock condition: Already high coverage AND future actions degrade or fail to improve
        if cov_now >= self.lock_threshold and peak_phi <= (phi_now + self.docking_drop_tolerance):
            self.last_decision = "LOCK"
            self.last_reason = f"Terminal peak reached (Cov: {cov_now*100:.1f}%, Phi: {phi_now:.3f}). Locked to prevent overshoot."
            return {
                "decision": "LOCK",
                "optimal_k": 0,
                "peak_cov": peak_cov,
                "peak_phi": peak_phi,
                "pred_covs": pred_covs,
                "pred_phis": pred_phis,
                "pred_poses": pred_poses,
                "actions_to_execute": np.zeros((0, 2), dtype=np.float32),
                "reason": self.last_reason,
            }

        # 2B. Truncate condition: Peak occurs in middle of horizon, trailing steps overshoot
        if best_k < (ta - 1) and pred_phis[-1] < (peak_phi - 0.02):
            self.last_decision = "TRUNCATE"
            self.last_reason = f"Overshoot detected! Peak Phi {peak_phi:.3f} (Cov {peak_cov*100:.1f}%) at step {best_k+1}/{ta}. Truncated trailing steps."
            return {
                "decision": "TRUNCATE",
                "optimal_k": best_k + 1,
                "peak_cov": peak_cov,
                "peak_phi": peak_phi,
                "pred_covs": pred_covs,
                "pred_phis": pred_phis,
                "pred_poses": pred_poses,
                "actions_to_execute": actions[: best_k + 1],
                "reason": self.last_reason,
            }

        # 2C. Improvement continuing towards peak
        self.last_decision = "EXECUTE"
        self.last_reason = f"Approaching goal docking (Cov: {peak_cov*100:.1f}%, Phi: {peak_phi:.3f})."
        return {
            "decision": "EXECUTE",
            "optimal_k": best_k + 1,
            "peak_cov": peak_cov,
            "peak_phi": peak_phi,
            "pred_covs": pred_covs,
            "pred_phis": pred_phis,
            "pred_poses": pred_poses,
            "actions_to_execute": actions[: best_k + 1],
            "reason": self.last_reason,
        }

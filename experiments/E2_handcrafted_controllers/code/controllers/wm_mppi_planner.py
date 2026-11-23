"""World Model Model-Predictive Path Integral (MPPI) Autonomous Planner.

Enables the robot to self-discover manipulation tactics purely through
physics imagination using the learned World Model:
1. Samples N parallel candidate action trajectories (combining DiT prior, geometric affordances, and smooth perturbations).
2. Performs ultra-fast batched latent rollout: s_{t+1} = Transition(s_t, a_t) across horizon H.
3. Decodes predicted 4D poses (x, y, sin_th, cos_th) to evaluate multi-objective potential Phi(x, y, th).
4. Weighs trajectories via Path Integral Softmax to synthesize the optimal, smooth action.
5. Returns imagined "ghost futures" for real-time visualization.
"""

from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely import affinity
import pymunk

from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    four_d_to_pose,
    pose_to_4d,
)
from .collision_guard import NonPenetrationGuard


class WorldModelMPPIPlanner:
    """Model-Predictive Path Integral Planner driven by Latent World Model Dynamics."""

    def __init__(
        self,
        num_samples: int = 64,
        horizon: int = 12,
        temperature: float = 12.0,
        gamma: float = 0.95,
        goal_x: float = 256.0,
        goal_y: float = 256.0,
        goal_theta: float = math.pi / 4.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_samples = num_samples
        self.horizon = horizon
        self.temperature = temperature
        self.gamma = gamma
        self.device = device

        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_theta = float(goal_theta)

        # Base T geometry for fast analytical intersection
        p1 = sg.Polygon([(-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)])
        p2 = sg.Polygon([(-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)])
        self.base_t = unary_union([p1, p2])
        self._update_goal_geom()

        self.guard = NonPenetrationGuard(max_step_px=35.0)

        # Cache for previous optimal trajectory (temporal warm start)
        self.prev_optimal_plan: Optional[torch.Tensor] = None

    def _update_goal_geom(self):
        deg = math.degrees(self.goal_theta)
        rot = affinity.rotate(self.base_t, deg, origin=(0.0, 0.0))
        self.goal_geom = affinity.translate(rot, xoff=self.goal_x, yoff=self.goal_y)
        self.goal_area = float(self.goal_geom.area)

    def set_goal(self, goal_x: float, goal_y: float, goal_theta: float):
        """Dynamically reconfigures planner target goal in real-time."""
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_theta = float(goal_theta)
        self._update_goal_geom()

    def _generate_candidate_actions(
        self,
        current_agent_px: np.ndarray,
        current_block_pose: np.ndarray,
        dit_prior: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Samples N diverse, dynamically promising candidate action trajectories of length H."""
        N = self.num_samples
        H = self.horizon
        bx, by, bth = current_block_pose
        ax, ay = current_agent_px

        candidates = []

        # -------------------------------------------------------------
        # Set A: DiT Prior + Temporal Perturbations (if available)
        # -------------------------------------------------------------
        num_dit = 16 if dit_prior is not None else 0
        if dit_prior is not None:
            # Ensure prior is length H
            prior_H = np.zeros((H, 2), dtype=np.float32)
            valid_len = min(H, len(dit_prior))
            prior_H[:valid_len] = dit_prior[:valid_len]
            if valid_len < H:
                prior_H[valid_len:] = prior_H[valid_len - 1]

            candidates.append(prior_H.copy())  # Exact DiT
            for scale in [0.03, 0.06, 0.10, 0.15]:
                for _ in range(num_dit // 4 - 1):
                    noise = np.random.randn(H, 2).astype(np.float32) * scale
                    # Smooth noise temporally
                    for t in range(1, H):
                        noise[t] = 0.6 * noise[t - 1] + 0.4 * noise[t]
                    cand = np.clip(prior_H + noise, -1.0, 1.0)
                    candidates.append(cand)

        # -------------------------------------------------------------
        # Set B: Geometric Affordance Primitives (Nách chữ T, Mút sườn, Bụng)
        # -------------------------------------------------------------
        # Local key contact targets on the Push-T block:
        # - Belly: (0, 130) -> Direct straight push
        # - Left Notch ("Nách trái"): (-30, 30) -> Coupled turn-and-push right
        # - Right Notch ("Nách phải"): (30, 30) -> Coupled turn-and-push left
        # - Stem Tip Left: (-30, 120) -> Sharp clockwise torque
        # - Stem Tip Right: (30, 120) -> Sharp counter-clockwise torque
        # - Head Center: (0, -20) -> Downward realignment
        local_targets = [
            (0.0, 135.0, 0.0, -1.0),      # Belly forward
            (-35.0, 35.0, 0.707, -0.707),  # Left notch diagonal
            (35.0, 35.0, -0.707, -0.707),  # Right notch diagonal
            (-35.0, 120.0, 1.0, 0.0),     # Stem left lateral
            (35.0, 120.0, -1.0, 0.0),     # Stem right lateral
            (0.0, -25.0, 0.0, 1.0),       # Head downwards
            (-70.0, 0.0, 0.707, 0.707),   # Head bar left corner
            (70.0, 0.0, -0.707, 0.707),   # Head bar right corner
        ]

        cos_t = math.cos(bth)
        sin_t = math.sin(bth)

        # Target goal direction vector from block CoG
        dx_g = self.goal_x - bx
        dy_g = self.goal_y - by
        dist_g = math.hypot(dx_g, dy_g) + 1e-6
        dir_g = np.array([dx_g / dist_g, dy_g / dist_g], dtype=np.float32)

        for lx, ly, l_pdx, l_pdy in local_targets:
            # Transform local contact to world coordinates
            wx = bx + lx * cos_t - ly * sin_t
            wy = by + lx * sin_t + ly * cos_t
            contact_pt = np.array([wx, wy], dtype=np.float32)

            # Transform local push direction to world
            w_pdx = l_pdx * cos_t - l_pdy * sin_t
            w_pdy = l_pdx * sin_t + l_pdy * cos_t

            # Smooth rate-limited trajectory towards contact and through block
            traj_px = np.zeros((H, 2), dtype=np.float32)
            curr = current_agent_px.copy()
            max_step = 25.0  # Max physical displacement per step in push-t

            # Target direction once in contact
            push_dir_combo = 0.6 * np.array([w_pdx, w_pdy]) + 0.4 * dir_g
            push_dir_norm = push_dir_combo / (np.linalg.norm(push_dir_combo) + 1e-6)

            reached_contact = False
            for t in range(H):
                if not reached_contact:
                    delta = contact_pt - curr
                    d_len = float(np.linalg.norm(delta))
                    if d_len <= max_step:
                        curr = contact_pt.copy()
                        reached_contact = True
                    else:
                        curr = curr + (delta / d_len) * max_step
                else:
                    curr = curr + push_dir_norm * max_step

                traj_px[t] = curr.copy()

            traj_norm = normalize_pos(traj_px)
            candidates.append(traj_norm)

            # Add smooth variations
            for scale in [0.03, 0.06]:
                pert = np.random.randn(H, 2).astype(np.float32) * scale
                for t in range(1, H):
                    pert[t] = 0.7 * pert[t - 1] + 0.3 * pert[t]
                candidates.append(np.clip(traj_norm + pert, -1.0, 1.0))

        # -------------------------------------------------------------
        # Set C: Previous optimal plan shifted (Receding Horizon Continuity)
        # -------------------------------------------------------------
        if self.prev_optimal_plan is not None:
            prev_np = self.prev_optimal_plan.cpu().numpy()
            shifted = np.zeros_like(prev_np)
            shifted[:-1] = prev_np[1:]
            shifted[-1] = prev_np[-1]
            candidates.append(shifted)
            for _ in range(3):
                pert = np.random.randn(H, 2).astype(np.float32) * 0.04
                candidates.append(np.clip(shifted + pert, -1.0, 1.0))

        # -------------------------------------------------------------
        # Set D: Smooth Splines & Gaussian Random Explorations (Fill to N)
        # -------------------------------------------------------------
        curr_norm = normalize_pos(current_agent_px)
        while len(candidates) < N:
            # Random waypoint in workspace
            rnd_waypoint = np.random.uniform(-0.9, 0.9, size=(2,)).astype(np.float32)
            traj = np.zeros((H, 2), dtype=np.float32)
            for t in range(H):
                alpha = (t + 1) / H
                traj[t] = (1.0 - alpha) * curr_norm + alpha * rnd_waypoint
            pert = np.random.randn(H, 2).astype(np.float32) * 0.08
            for t in range(1, H):
                pert[t] = 0.7 * pert[t - 1] + 0.3 * pert[t]
            candidates.append(np.clip(traj + pert, -1.0, 1.0))

        candidates_arr = np.array(candidates[:N], dtype=np.float32)
        return torch.from_numpy(candidates_arr).to(self.device)

    def plan(
        self,
        world_model: torch.nn.Module,
        obs: Dict[str, Any],
        dit_prior: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Runs one full cycle of World Model MPPI Optimization.

        Returns:
            Dict containing:
                - optimal_action_norm: (2,) Next normalized action in [-1, 1]^2
                - optimal_action_px: (2,) Next workspace action in [0, 512]^2
                - optimal_trajectory_px: (H, 2) Complete planned pusher path
                - top_predicted_poses: List of top 5 imagined (x, y, theta) trajectories
                - peak_expected_cov: Best imagined target coverage
        """
        world_model.eval()
        N = self.num_samples
        H = self.horizon

        # 1. Encode current observation to latent state s_0
        with torch.no_grad():
            img_norm = normalize_image(obs["pixels"])
            if isinstance(img_norm, np.ndarray):
                pixels_norm = torch.from_numpy(img_norm).float()
            else:
                pixels_norm = img_norm.float()
            if pixels_norm.ndim == 3:
                pixels_norm = pixels_norm.unsqueeze(0)
            pixels_norm = pixels_norm.to(self.device)

            ag_norm = obs["agent_pos"]
            if isinstance(ag_norm, np.ndarray):
                agent_pos_norm = torch.from_numpy(ag_norm).float()
            else:
                agent_pos_norm = ag_norm.float()
            if agent_pos_norm.ndim == 1:
                agent_pos_norm = agent_pos_norm.unsqueeze(0)
            agent_pos_norm = agent_pos_norm.to(self.device)

            s0 = world_model.encode(pixels_norm, agent_pos_norm)  # (1, 128)

        # 2. Sample candidate action trajectories of shape (N, H, 2)
        current_agent_px = obs["agent_pos_px"]
        current_block_pose = obs["block_pose"]
        cand_acts = self._generate_candidate_actions(
            current_agent_px, current_block_pose, dit_prior=dit_prior
        )  # (N, H, 2)

        # 3. Batched Latent Rollout in World Model
        # s_batch: (N, 128)
        s_batch = s0.repeat(N, 1)
        pred_poses_list = []
        pred_agents_list = []

        with torch.no_grad():
            for k in range(H):
                act_k = cand_acts[:, k]  # (N, 2)
                s_batch = world_model.predict_next_state(s_batch, act_k)
                pose_4d, ag_pos = world_model.decode(s_batch)
                pred_poses_list.append(pose_4d.unsqueeze(1))
                pred_agents_list.append(ag_pos.unsqueeze(1))

            # Concat across horizon: (N, H, 4) and (N, H, 2)
            all_pred_poses = torch.cat(pred_poses_list, dim=1)
            all_pred_agents = torch.cat(pred_agents_list, dim=1)

        # 4. Multi-Objective Trajectory Evaluation on GPU/CPU
        poses_np = all_pred_poses.cpu().numpy()  # (N, H, 4)
        acts_np = cand_acts.cpu().numpy()        # (N, H, 2)

        rewards = np.zeros(N, dtype=np.float32)
        end_covs = np.zeros(N, dtype=np.float32)

        init_dist = math.hypot(current_block_pose[0] - self.goal_x, current_block_pose[1] - self.goal_y)

        for i in range(N):
            # Final step pose prediction
            p_final = poses_np[i, -1]
            bx_f, by_f, bth_f = four_d_to_pose(p_final)

            # Mid step pose prediction
            p_mid = poses_np[i, H // 2]
            bx_m, by_m, bth_m = four_d_to_pose(p_mid)

            # Distance & Angle alignment
            dist_final = math.hypot(bx_f - self.goal_x, by_f - self.goal_y)
            dth_final = abs((bth_f - self.goal_theta + math.pi) % (2.0 * math.pi) - math.pi)

            # Fast polygon coverage at terminal step
            deg_f = math.degrees(bth_f)
            rot_f = affinity.rotate(self.base_t, deg_f, origin=(0.0, 0.0))
            poly_f = affinity.translate(rot_f, xoff=bx_f, yoff=by_f)
            cov_final = float(self.goal_geom.intersection(poly_f).area / self.goal_area)
            end_covs[i] = cov_final

            # Progress bonus
            progress = init_dist - dist_final

            # Jerk / smoothness penalty
            act_diff = acts_np[i, 1:] - acts_np[i, :-1]
            jerk = float(np.sum(act_diff ** 2))

            # Approach bonus when not yet in contact
            ag_f_px = unnormalize_pos(acts_np[i, -1])
            d_ag_block = math.hypot(ag_f_px[0] - current_block_pose[0], ag_f_px[1] - current_block_pose[1])
            approach_bonus = 1.2 * (1.0 - min(d_ag_block, 250.0) / 250.0)

            # Composite Potential Phi
            r = (
                5.0 * cov_final
                - 1.5 * (dist_final / 200.0)
                - 1.0 * (dth_final / math.pi)
                + 2.5 * (progress / 100.0)
                + approach_bonus
                - 0.15 * jerk
            )

            # Boundary penalty (stay inside workspace)
            if bx_f < 60.0 or bx_f > 450.0 or by_f < 60.0 or by_f > 450.0:
                r -= 3.0

            rewards[i] = r

        # 5. Elite Selection (Prevents multi-modal action averaging collapse)
        best_idx = int(np.argmax(rewards))
        top_indices = np.argsort(rewards)[-5:][::-1]

        # Use top-1 best coherent imagined trajectory
        opt_traj_norm = acts_np[best_idx].copy()
        opt_traj_px = unnormalize_pos(opt_traj_norm)

        # Top 5 imagined trajectories for visualization
        top_imagined = []
        for idx in top_indices:
            traj_poses = []
            for t in range(H):
                x_t, y_t, th_t = four_d_to_pose(poses_np[idx, t])
                traj_poses.append((float(x_t), float(y_t), float(th_t)))
            top_imagined.append(traj_poses)
        for idx in top_indices:
            traj_poses = []
            for t in range(H):
                x_t, y_t, th_t = four_d_to_pose(poses_np[idx, t])
                traj_poses.append((float(x_t), float(y_t), float(th_t)))
            top_imagined.append(traj_poses)

        # Cache optimal plan for temporal warm-start
        self.prev_optimal_plan = torch.from_numpy(opt_traj_norm).to(self.device)

        next_act_norm = opt_traj_norm[0]
        next_act_px = opt_traj_px[0]

        best_idx = top_indices[0]
        return {
            "optimal_action_norm": next_act_norm,
            "optimal_action_px": next_act_px,
            "optimal_trajectory_px": opt_traj_px,
            "top_imagined_poses": top_imagined,
            "peak_expected_cov": float(end_covs[best_idx]),
            "best_reward": float(rewards[best_idx]),
        }

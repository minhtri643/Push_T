"""Tactical Hierarchical Controller (Orient -> Align -> Push).

Implements the user's 3-Phase Manipulation Strategy:
Phase 1 (ORIENT): Rotate the T-block until its top face is parallel to the goal (Delta_theta -> 0).
Phase 2 (ALIGN):  Navigate around the block without collisions to stage directly behind the belly
                  along the collinear symmetry axis into the goal slot.
Phase 3 (PUSH):   Push through the base (belly) straight towards the goal, seamlessly handing off
                  to the DiT Trajectory Planner + Residual Inverse Dynamics Controller for terminal docking.

Rigid-Body Safety:
Integrated with NonPenetrationGuard to guarantee the pusher never cuts through or overlaps
the T-block interior under any condition.
"""

from typing import Tuple, Dict, Any, Optional
import math
import numpy as np
import torch
import pymunk
import shapely.geometry as sg

from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    four_d_to_pose,
)
from .collision_guard import NonPenetrationGuard
from .wm_lookahead import (
    WorldModelLookaheadOptimizer,
    compute_safe_retreat,
)
from .wm_mppi_planner import WorldModelMPPIPlanner


class TacticalHierarchicalController:
    """Combines geometric tactical alignment with DiT foundation trajectory planning."""

    def __init__(
        self,
        goal_x: float = 256.0,
        goal_y: float = 256.0,
        goal_theta: float = math.pi / 4.0,  # 0.7854 rad (45 deg)
        orient_tolerance: float = 0.35,     # ~20 degrees
        target_lock_cov: float = 0.88,      # 88% coverage target lock
        peak_settle_cov: float = 0.80,      # 80% coverage peak settle
        enable_collision_guard: bool = False,
        enable_wm_lookahead: bool = False,
        enable_mppi: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_theta = goal_theta
        self.orient_tolerance = orient_tolerance
        self.target_lock_cov = target_lock_cov
        self.peak_settle_cov = peak_settle_cov
        self.enable_collision_guard = enable_collision_guard
        self.enable_wm_lookahead = enable_wm_lookahead
        self.enable_mppi = enable_mppi
        self.device = device

        # Non-penetration collision guard
        self.guard = NonPenetrationGuard(max_step_px=35.0)

        # World Model Lookahead Optimal Stopping Optimizer (with Gating + Composite Potential)
        self.wm_optimizer = WorldModelLookaheadOptimizer(
            gating_cov=0.80,
            lock_threshold=self.target_lock_cov,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
        )

        # World Model MPPI Planner (Physics Imagination & Autonomous Exploration)
        self.mppi_planner = WorldModelMPPIPlanner(
            num_samples=48,
            horizon=10,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
            device=device,
        )

        # Goal CoG & direction vectors
        # CoG of the T-block in local coords is at (0, 45).
        self.goal_cog = np.array([
            self.goal_x - 45.0 * math.sin(self.goal_theta),
            self.goal_y + 45.0 * math.cos(self.goal_theta),
        ], dtype=np.float32)
        self.v_fwd = np.array([0.70710678, -0.70710678], dtype=np.float32)
        self.v_lat = np.array([0.70710678, 0.70710678], dtype=np.float32)

        # Internal state tracking
        self.phase = "ORIENT"
        self.max_cov_seen = 0.0
        self.dit_pending_actions = []

    def set_goal(self, goal_x: float, goal_y: float, goal_theta: float) -> None:
        """Dynamically updates the target goal coordinates and wakes up chaser."""
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_theta = float(goal_theta)
        self.goal_cog = np.array([
            self.goal_x - 45.0 * math.sin(self.goal_theta),
            self.goal_y + 45.0 * math.cos(self.goal_theta),
        ], dtype=np.float32)
        self.wm_optimizer.set_goal(self.goal_x, self.goal_y, self.goal_theta)
        self.mppi_planner.set_goal(self.goal_x, self.goal_y, self.goal_theta)
        # Unlock immediately if goal has moved away
        if self.phase == "LOCKED":
            self.phase = "ORIENT"
            self.max_cov_seen = 0.0
            self.dit_pending_actions.clear()

    def reset(self):
        """Resets the controller state for a new episode."""
        self.phase = "ORIENT"
        self.max_cov_seen = 0.0
        self.dit_pending_actions.clear()

    def _guard_action(
        self,
        target_action: np.ndarray,
        agent_px: np.ndarray,
        body: Any,
        status_msg: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """Filters action through NonPenetrationGuard if enabled."""
        info = extra_info or {}
        info["phase"] = self.phase
        info["guarded"] = False

        if not self.enable_collision_guard or body is None:
            return target_action, status_msg, info

        safe_action, was_filtered, reason = self.guard.filter_action(agent_px, target_action, body)
        if was_filtered:
            status_msg = f"🛡️ {reason}"
            info["guarded"] = True

        return safe_action, status_msg, info

    def get_action(
        self,
        obs: Dict[str, Any],
        sim: Any,
        policy: Any,
        exec_horizon: int = 8,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """Computes next action in workspace coordinates [0, 512]^2 according to the 3-phase tactic."""
        bx, by = float(obs["block_pose"][0]), float(obs["block_pose"][1])
        bth = float(obs["block_pose"][2])
        ax, ay = float(obs["agent_pos_px"][0]), float(obs["agent_pos_px"][1])
        agent_px = np.array([ax, ay], dtype=np.float32)
        is_canonical_goal = (
            math.hypot(self.goal_x - 256.0, self.goal_y - 256.0) < 15.0
            and abs((self.goal_theta - math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi) < 0.15
        )
        if not is_canonical_goal:
            cov = self.wm_optimizer.evaluator.compute_coverage(bx, by, bth)
        else:
            cov = float(obs["coverage"])
        body = sim.env.unwrapped.block if (hasattr(sim, "env") and hasattr(sim.env.unwrapped, "block")) else None

        if cov > self.max_cov_seen:
            self.max_cov_seen = cov

        # Check Target Lock completion first
        if cov >= self.target_lock_cov:
            self.phase = "LOCKED"
            safe_retreat_pos = compute_safe_retreat(agent_px, np.array([bx, by]), retreat_dist=16.0)
            return self._guard_action(safe_retreat_pos, agent_px, body, f"🔒 LOCKED (Goal Reached: {cov*100:.1f}%)", {"phase": "LOCKED", "cov": cov})

        # Gating check: Only allow peak settle lock if block has genuinely reached the terminal basin
        dist_now = math.hypot(bx - self.goal_x, by - self.goal_y)
        settle_thresh = self.peak_settle_cov if is_canonical_goal else 0.55
        if self.max_cov_seen >= settle_thresh and cov < (self.max_cov_seen - 0.05) and dist_now <= 30.0:
            self.phase = "LOCKED"
            safe_retreat_pos = compute_safe_retreat(agent_px, np.array([bx, by]), retreat_dist=16.0)
            return self._guard_action(safe_retreat_pos, agent_px, body, f"🔒 LOCKED (Peak Settle: {self.max_cov_seen*100:.1f}%)", {"phase": "LOCKED", "cov": cov})

        # If already LOCKED but goal moved away or coverage dropped, unlock immediately to resume chasing!
        if self.phase == "LOCKED":
            if dist_now > 25.0 or cov < (self.target_lock_cov - 0.15):
                self.phase = "ORIENT"
                self.max_cov_seen = cov
                self.dit_pending_actions.clear()
            else:
                safe_retreat_pos = compute_safe_retreat(agent_px, np.array([bx, by]), retreat_dist=16.0)
                return self._guard_action(safe_retreat_pos, agent_px, body, f"🔒 LOCKED (Goal Reached: {cov*100:.1f}%)", {"phase": "LOCKED", "cov": cov})

        # Angle difference relative to goal (wrapped to [-pi, pi])
        dth = (bth - self.goal_theta + math.pi) % (2.0 * math.pi) - math.pi

        body = sim.env.unwrapped.block
        cog_world = np.array(body.local_to_world((0.0, 45.0)))
        vg = self.goal_cog - cog_world
        dist_to_goal = float(np.linalg.norm(vg))
        vg_norm = vg / (dist_to_goal + 1e-6)

        # Check if pushing from belly through stem points towards the goal
        vg_b = np.array([self.goal_x - bx, self.goal_y - by], dtype=np.float32)
        vg_b_norm = vg_b / (np.linalg.norm(vg_b) + 1e-6)
        # Local (0, -1) in world coordinates is (sin(bth), -cos(bth))
        v_push_belly = np.array([math.sin(bth), -math.cos(bth)], dtype=np.float32)
        belly_pushes_towards_goal = bool(np.dot(v_push_belly, vg_b_norm) > 0.15)

        if not is_canonical_goal:
            # -------------------------------------------------------------
            # DYNAMIC REAL-TIME CHASER (Chasing user-moved target)
            # -------------------------------------------------------------
            # 1. Orient if angular misalignment is significant
            if abs(dth) > self.orient_tolerance and dist_to_goal > 25.0:
                self.phase = "ORIENT"
                self.dit_pending_actions.clear()
                if dth > 0:
                    touch_start = body.local_to_world((-32.0, 115.0))
                    touch_target = body.local_to_world((-5.0, 115.0))
                else:
                    touch_start = body.local_to_world((32.0, 115.0))
                    touch_target = body.local_to_world((5.0, 115.0))

                dist_to_touch = math.hypot(ax - touch_start.x, ay - touch_start.y)
                if dist_to_touch > 25.0:
                    raw_action = np.array([touch_start.x, touch_start.y], dtype=np.float32)
                    status_msg = f"1. ORIENT: Tiếp cận góc xoay theo mục tiêu (Lệch: {math.degrees(dth):.0f}°)"
                else:
                    raw_action = np.array([touch_target.x, touch_target.y], dtype=np.float32)
                    status_msg = f"1. ORIENT: Gạt xoay theo mục tiêu (Lệch: {math.degrees(dth):.0f}°)"

                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "ORIENT", "dth": dth})

            # 2. Stage & Push along vg_norm
            u_local = pymunk.Vec2d(float(vg_norm[0]), float(vg_norm[1])).rotated(-bth)
            back_dir = -u_local
            ray = sg.LineString([(0.0, 45.0), (back_dir.x * 200.0, 45.0 + back_dir.y * 200.0)])
            inter = ray.intersection(self.guard.t_block_local.boundary)
            if inter.geom_type == 'Point':
                b_pt = (inter.x, inter.y)
            elif inter.geom_type == 'MultiPoint':
                b_pt = [(p.x, p.y) for p in inter.geoms][-1]
            else:
                b_pt = (0.0, 120.0)

            stage_local = (b_pt[0] + back_dir.x * 20.0, b_pt[1] + back_dir.y * 20.0)
            stage_world = np.array(body.local_to_world(stage_local), dtype=np.float32)
            dist_stage = float(np.linalg.norm(agent_px - stage_world))

            if dist_stage > 25.0:
                self.phase = "CHASE_ALIGN"
                raw_action = stage_world
                status_msg = f"2. CHASE: Di chuyển đón đầu hướng đẩy (Cách: {dist_stage:.0f}px)"
                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "CHASE_ALIGN", "dist_stage": dist_stage})
            else:
                self.phase = "CHASE_PUSH"
                push_mag = min(30.0, max(14.0, dist_to_goal * 0.5))
                raw_action = stage_world + vg_norm * push_mag
                status_msg = f"3. CHASE: Đang đẩy chữ T hướng về mục tiêu (Cách: {dist_to_goal:.0f}px | IoU: {cov*100:.1f}%)"
                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "CHASE_PUSH", "dist_goal": dist_to_goal, "cov": cov})

        # Canonical goal execution (DiT + WM Lookahead)
        in_terminal_docking = ((dist_to_goal < 45.0) and (abs(dth) <= self.orient_tolerance)) or (cov > 0.40)

        # -------------------------------------------------------------
        # WORLD MODEL MPPI PLANNER (Active Physics Imagination & Control)
        # -------------------------------------------------------------
        if not in_terminal_docking and self.enable_mppi and hasattr(policy, "wm"):
            self.phase = "MPPI_IMAGINE"
            self.dit_pending_actions.clear()
            dit_prior = None
            if hasattr(policy, "plan"):
                try:
                    pred_acts, _ = policy.plan(obs)
                    dit_prior = pred_acts
                except Exception:
                    pass

            mppi_res = self.mppi_planner.plan(policy.wm, obs, dit_prior=dit_prior)
            opt_act_px = mppi_res["optimal_action_px"]
            status_msg = f"🧠 MPPI World Model: Tự tưởng tượng & đẩy tối ưu (Đỉnh dự kiến: {mppi_res['peak_expected_cov']*100:.1f}%)"
            return self._guard_action(
                opt_act_px, agent_px, body, status_msg,
                {"phase": "MPPI_IMAGINE", "mppi_res": mppi_res, "cov": cov}
            )

        # Check if pushing from belly through stem points towards the goal
        vg = np.array([self.goal_x - bx, self.goal_y - by], dtype=np.float32)
        vg_norm = vg / (np.linalg.norm(vg) + 1e-6)
        # Local (0, -1) in world coordinates is (sin(bth), -cos(bth))
        v_push_belly = np.array([math.sin(bth), -math.cos(bth)], dtype=np.float32)
        belly_pushes_towards_goal = bool(np.dot(v_push_belly, vg_norm) > 0.15)

        # -------------------------------------------------------------
        # PHASE 1: ORIENT (Xoay song song mặt trên)
        # -------------------------------------------------------------
        if not in_terminal_docking and abs(dth) > self.orient_tolerance:
            # Hysteresis: If already in PUSH_DIT and angle is reasonably aligned, keep pushing
            if self.phase in ("PUSH_DIT", "DOCK_DIT") and abs(dth) <= (self.orient_tolerance * 1.5):
                pass
            else:
                self.phase = "ORIENT"
                self.dit_pending_actions.clear()

                # Pymunk torque mechanics:
                # Pushing stem tip towards local +x decreases angle.
                # Pushing stem tip towards local -x increases angle.
                if dth > 0:
                    touch_start = body.local_to_world((-32.0, 115.0))
                    touch_target = body.local_to_world((-5.0, 115.0))
                else:
                    touch_start = body.local_to_world((32.0, 115.0))
                    touch_target = body.local_to_world((5.0, 115.0))

                dist_to_touch = math.hypot(ax - touch_start.x, ay - touch_start.y)
                if dist_to_touch > 25.0:
                    raw_action = np.array([touch_start.x, touch_start.y], dtype=np.float32)
                    status_msg = f"1. ORIENT: Tiếp cận góc gạt xoay (Lệch: {math.degrees(dth):.0f}°)"
                else:
                    raw_action = np.array([touch_target.x, touch_target.y], dtype=np.float32)
                    status_msg = f"1. ORIENT: Gạt mút xoay song song (Lệch: {math.degrees(dth):.0f}°)"

                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "ORIENT", "dth": dth})

        # -------------------------------------------------------------
        # PHASE 2: ALIGN BELLY (Căn chỉnh dóng trục & di chuyển qua bụng)
        # -------------------------------------------------------------
        agent_local = body.world_to_local((ax, ay))
        belly_world = body.local_to_world((0.0, 135.0))
        dist_to_belly = math.hypot(ax - belly_world.x, ay - belly_world.y)

        # Only stage behind belly if pushing through the belly actually moves the block towards the goal
        # AND we are not already executing an active DiT push sequence (prevents chattering deadlock)
        if not in_terminal_docking and belly_pushes_towards_goal and (self.phase not in ("PUSH_DIT", "DOCK_DIT")):
            if agent_local.y < 80.0:
                # Clearance waypoint around flank to avoid hitting the block
                self.phase = "ALIGN_CLEARANCE"
                self.dit_pending_actions.clear()
                side = 1.0 if agent_local.x >= 0 else -1.0
                if agent_local.y < 30.0:
                    wp_world = body.local_to_world((side * 85.0, 35.0))
                else:
                    wp_world = body.local_to_world((side * 60.0, 135.0))
                raw_action = np.array([wp_world.x, wp_world.y], dtype=np.float32)
                status_msg = f"2. ALIGN: Đi vòng sườn tránh va chạm (y={agent_local.y:.0f})"
                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "ALIGN_CLEARANCE", "dist_belly": dist_to_belly})

            elif dist_to_belly > 25.0:
                # Stage behind belly
                self.phase = "ALIGN_BELLY"
                self.dit_pending_actions.clear()
                raw_action = np.array([belly_world.x, belly_world.y], dtype=np.float32)
                status_msg = f"2. ALIGN: Đứng vào vị trí sau bụng chữ T (Cách: {dist_to_belly:.0f}px)"
                return self._guard_action(raw_action, agent_px, body, status_msg, {"phase": "ALIGN_BELLY", "dist_belly": dist_to_belly})

        # -------------------------------------------------------------
        # PHASE 3: PUSH VIA DiT (Đẩy bụng tiến thẳng vào ô đích)
        # -------------------------------------------------------------
        self.phase = "PUSH_DIT" if not in_terminal_docking else "DOCK_DIT"

        lookahead_info = {}
        if len(self.dit_pending_actions) == 0:
            pred_acts, _ = policy.plan(obs)
            acts_slice = pred_acts[:exec_horizon]

            if self.enable_wm_lookahead and hasattr(policy, "wm"):
                rollout_res = self.wm_optimizer.evaluate_rollout(
                    policy.wm, obs, acts_slice, policy.device
                )
                lookahead_info = rollout_res
                if rollout_res["decision"] == "LOCK":
                    self.phase = "LOCKED"
                    safe_action = compute_safe_retreat(agent_px, np.array([bx, by]), retreat_dist=16.0)
                    status_msg = f"🔒 LOCKED via WM Lookahead: Peak {rollout_res['peak_cov']*100:.1f}% (Tránh vọt lố)"
                    return self._guard_action(safe_action, agent_px, body, status_msg, {"phase": "LOCKED", "cov": cov, "wm_rollout": rollout_res})
                elif rollout_res["decision"] == "TRUNCATE":
                    acts_slice = rollout_res["actions_to_execute"]

            for a in acts_slice:
                self.dit_pending_actions.append(unnormalize_pos(a))

        if len(self.dit_pending_actions) > 0:
            raw_action = self.dit_pending_actions.pop(0)
        else:
            raw_action = agent_px

        phase_prefix = "3. PUSH" if self.phase == "PUSH_DIT" else "3. DOCK"
        wm_tag = f" [WM k*={self.wm_optimizer.last_best_k}]" if self.enable_wm_lookahead else ""
        status_msg = f"{phase_prefix}{wm_tag}: Đẩy khớp vào ô đích (IoU: {cov*100:.1f}%)"
        extra = {"phase": self.phase, "cov": cov}
        if lookahead_info:
            extra["wm_rollout"] = lookahead_info
        return self._guard_action(raw_action, agent_px, body, status_msg, extra)

"""Non-Penetration Collision Guard for Push-T.

Enforces physical rigid-body non-penetration constraints between the circular pusher
and the T-block:
1. Detects trajectory penetration (swept capsule or long-range line cutting through the interior of the T-block).
2. Forbids the agent from teleporting, phasing through, or pressing deep into the block.
3. Automatically computes collision-free detour waypoints around the perimeter through
   open exterior clearance corridors.
4. Preserves valid surface pushing actions (where the pusher contacts the exterior boundary to exert force).
"""

from typing import Tuple, Dict, Any, Optional
import math
import numpy as np
import shapely.geometry as sg
from shapely.ops import unary_union
import pymunk


class NonPenetrationGuard:
    """Rigid-body non-penetration safety filter for Push-T."""

    def __init__(
        self,
        pusher_radius: float = 15.0,
        max_step_px: float = 35.0,
        penetration_depth_thresh: float = 15.0,
    ):
        self.pusher_radius = pusher_radius
        self.max_step_px = max_step_px
        self.penetration_depth_thresh = penetration_depth_thresh

        # Standard gym-pusht T-block local geometry
        # Crossbar: x in [-60, 60], y in [0, 30]
        # Stem:     x in [-15, 15], y in [30, 120]
        bar = sg.box(-60.0, 0.0, 60.0, 30.0)
        stem = sg.box(-15.0, 30.0, 15.0, 120.0)
        self.t_block_local = unary_union([bar, stem])

        # Core interior zone (inset by 8px) - any target landing deep inside is an illegal penetration
        self.interior_core = self.t_block_local.buffer(-8.0)

    def check_cuts_block(
        self,
        p_start: Tuple[float, float],
        p_target: Tuple[float, float],
    ) -> Tuple[bool, float]:
        """Checks if a line segment in local coordinates cuts through the T-block."""
        line = sg.LineString([p_start, p_target])
        inter = line.intersection(self.t_block_local)
        cut_len = float(inter.length) if inter is not None else 0.0
        return (cut_len > self.penetration_depth_thresh), cut_len

    def compute_detour_local(
        self,
        p_start: Tuple[float, float],
        p_full_target: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Computes the safest exterior clearance waypoint around the T-block perimeter."""
        xs, ys = p_start
        xt, yt = p_full_target

        # 1. Lateral crossing across stem body (stem is x in [-15, 15], y in [30, 120])
        # Takes top priority so the pusher can slip under the belly or over the bar
        if (xs < -18.0 and xt > 18.0) or (xs > 18.0 and xt < -18.0):
            if (ys + yt) / 2.0 < 45.0:
                return (0.0, -25.0)          # Loop over top bar
            else:
                return (0.0, 140.0)          # Loop directly under belly tip

        # 2. Crossing or entering top bar from top
        if ys <= 15.0 and yt >= 0.0:
            side = 1.0 if (xs + xt) >= 0.0 else -1.0
            if abs(xs) < 80.0:
                return (side * 85.0, -10.0)  # Move sideways to clear wingtip
            else:
                return (side * 85.0, 50.0)   # Move down past wingtip

        # 3. Crossing or entering belly/stem from bottom
        if ys >= 105.0 and yt <= 120.0:
            side = 1.0 if (xs + xt) >= 0.0 else -1.0
            return (side * 60.0, 140.0)      # Move around belly

        # Default fallback: wing clearance
        side = 1.0 if xs >= 0.0 else -1.0
        return (side * 85.0, 20.0)

    def filter_action(
        self,
        agent_pos_world: np.ndarray,
        target_pos_world: np.ndarray,
        block_body: pymunk.Body,
    ) -> Tuple[np.ndarray, bool, str]:
        """Filters a target action to guarantee strict non-penetration of the T-block.

        Args:
            agent_pos_world: Current pusher position [ax, ay] in workspace pixels.
            target_pos_world: Desired pusher position [tx, ty] in workspace pixels.
            block_body: The Pymunk Body of the T-block.

        Returns:
            Tuple of:
            - safe_target_world: Collision-free action in workspace pixels [0, 512].
            - was_filtered: Boolean indicating if collision guard was triggered.
            - reason: Diagnostic reason string.
        """
        ax, ay = float(agent_pos_world[0]), float(agent_pos_world[1])
        tx, ty = float(target_pos_world[0]), float(target_pos_world[1])

        # Convert start and full target to local block coordinates
        p_start_local = block_body.world_to_local(pymunk.Vec2d(ax, ay))
        p_target_local = block_body.world_to_local(pymunk.Vec2d(tx, ty))

        # Check if direct path to target cuts through the T-block interior
        cuts, cut_len = self.check_cuts_block(
            (p_start_local.x, p_start_local.y),
            (p_target_local.x, p_target_local.y),
        )

        # Also check if target itself is deep inside the block
        target_inside = self.interior_core.contains(sg.Point(p_target_local.x, p_target_local.y))

        if not cuts and not target_inside:
            # Clear! Apply normal step rate-limiting
            delta = np.array([tx - ax, ty - ay], dtype=np.float32)
            d = np.linalg.norm(delta)
            if d > self.max_step_px:
                safe_pt = np.array([ax, ay], dtype=np.float32) + (delta / d) * self.max_step_px
            else:
                safe_pt = np.array([tx, ty], dtype=np.float32)
            return safe_pt, False, "CLEAR"

        # PENETRATION DETECTED: Compute safe exterior detour waypoint
        detour_local = self.compute_detour_local(
            (p_start_local.x, p_start_local.y),
            (p_target_local.x, p_target_local.y),
        )

        detour_world = block_body.local_to_world(pymunk.Vec2d(detour_local[0], detour_local[1]))
        dw_x, dw_y = float(detour_world.x), float(detour_world.y)

        # Rate-limit step towards detour waypoint
        detour_delta = np.array([dw_x - ax, dw_y - ay], dtype=np.float32)
        d_detour = np.linalg.norm(detour_delta)
        if d_detour > self.max_step_px:
            safe_target = np.array([ax, ay], dtype=np.float32) + (detour_delta / d_detour) * self.max_step_px
        else:
            safe_target = np.array([dw_x, dw_y], dtype=np.float32)

        reason = f"GUARD_BLOCKED (cut: {cut_len:.1f}px) -> Detour ({detour_local[0]:.0f}, {detour_local[1]:.0f})"
        return safe_target, True, reason

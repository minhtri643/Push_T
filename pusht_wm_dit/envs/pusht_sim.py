"""Push-T Simulation Environment Wrapper and Video Recorder.

Encapsulates gym-pusht / Pymunk physics with:
1. RGB frame observations (96x96x3) and agent end-effector position (x_agent, y_agent).
2. Ground-truth T-block pose extraction (x_T, y_T, theta_T) and 4D continuous vector [x_T, y_T, sin(theta), cos(theta)].
3. Target Coverage IoU tracking.
4. Video recording with telemetry HUD overlay.
5. Algorithmic expert demonstrator.
"""

from typing import Dict, Any, Tuple, Optional, Union
import os
import cv2
import numpy as np
import gymnasium as gym
import gym_pusht
import imageio

from pusht_wm_dit.utils.normalizer import (
    normalize_pos,
    unnormalize_pos,
    pose_to_4d,
    WORKSPACE_MIN,
    WORKSPACE_MAX,
)


class PushTSim:
    """Production wrapper for gym-pusht Pymunk simulation."""

    def __init__(
        self,
        env_name: str = "gym_pusht/PushT-v0",
        render_size: Tuple[int, int] = (96, 96),
        max_steps: int = 200,
        render_mode: str = "rgb_array",
        goal_pose: Tuple[float, float, float] = (256.0, 256.0, np.pi / 4.0),
    ):
        self.env_name = env_name
        self.render_size = render_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.goal_pose = np.array(goal_pose, dtype=np.float32)

        self.env = gym.make(
            env_name,
            obs_type="pixels_agent_pos",
            render_mode=render_mode,
        )
        self.current_step = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_info: Optional[Dict[str, Any]] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment."""
        obs, info = self.env.reset(seed=seed)
        if hasattr(self.env.unwrapped, "goal_pose"):
            self.env.unwrapped.goal_pose = self.goal_pose
        self.current_step = 0
        self._last_obs = obs
        self._last_info = info
        processed_obs = self._with_history(self._format_observation(obs, info), None)
        return processed_obs, info

    def _with_history(self, obs: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adds the previous step's frame/pusher position and the pusher velocity (per step, normalized).
        After a reset or teleport there is no previous step, so the current one is repeated."""
        prev = prev or obs
        obs["prev_pixels"] = prev["pixels"]
        obs["prev_agent_pos"] = prev["agent_pos"]
        obs["agent_vel"] = (np.asarray(obs["agent_pos"]) - np.asarray(prev["agent_pos"])).astype(np.float32)
        self._cur = obs
        return obs

    def set_goal_pose(self, goal_pose: Union[np.ndarray, list, tuple]) -> None:
        """Updates the target goal pose [gx, gy, gtheta] in gym-pusht."""
        self.goal_pose = np.array(goal_pose, dtype=np.float32)
        if hasattr(self.env.unwrapped, "goal_pose"):
            self.env.unwrapped.goal_pose = self.goal_pose

    def step(
        self,
        action: Union[np.ndarray, list],
        normalized_action: bool = True,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Executes a low-level action in the Push-T environment.

        Args:
            action: Command [x_cmd, y_cmd]. Can be in [-1, 1] or [0, 512].
            normalized_action: If True, action is in [-1, 1] and will be unnormalized.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        act = np.asarray(action, dtype=np.float32).flatten()
        if normalized_action:
            act_sim = unnormalize_pos(act)
        else:
            act_sim = act

        act_sim = np.clip(act_sim, WORKSPACE_MIN, WORKSPACE_MAX)

        obs, reward, terminated, truncated, info = self.env.step(act_sim)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        self._last_obs = obs
        self._last_info = info
        processed_obs = self._with_history(self._format_observation(obs, info), getattr(self, "_cur", None))
        return processed_obs, float(reward), bool(terminated), bool(truncated), info

    def _format_observation(
        self,
        raw_obs: Dict[str, Any],
        raw_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Formats observations into unified format."""
        # RGB frame: (96, 96, 3) uint8
        pixels = raw_obs["pixels"]
        if pixels.shape[:2] != self.render_size:
            pixels = cv2.resize(pixels, self.render_size, interpolation=cv2.INTER_AREA)

        # Agent position: [0, 512] -> [-1, 1]
        agent_pos_px = np.asarray(raw_info.get("pos_agent", raw_obs.get("agent_pos")), dtype=np.float32)
        agent_pos_norm = normalize_pos(agent_pos_px)

        # T-block pose: [x_T, y_T, theta_T]
        block_pose = np.asarray(raw_info["block_pose"], dtype=np.float32)
        bx, by, btheta = float(block_pose[0]), float(block_pose[1]), float(block_pose[2])

        # 4D continuous pose: [x_norm, y_norm, sin(theta), cos(theta)] in [-1, 1]^4
        block_pose_4d = pose_to_4d(bx, by, btheta)

        # Coverage = intersection / goal_area, exactly as gym-pusht computes reward & success.
        # (After _set_state the info dict has no "coverage" key, so query the env directly
        # instead of falling back to a different metric such as IoU.)
        coverage = raw_info.get("coverage")
        if coverage is None:
            coverage = self.env.unwrapped._get_coverage()
        coverage = float(coverage)

        return {
            "pixels": pixels,                   # (96, 96, 3) uint8
            "agent_pos": agent_pos_norm,         # (2,) [-1, 1]
            "agent_pos_px": agent_pos_px,       # (2,) [0, 512]
            "block_pose": block_pose,           # (3,) [x, y, theta]
            "block_pose_4d": block_pose_4d,     # (4,) [x_norm, y_norm, sin(th), cos(th)]
            "coverage": coverage,               # float [0.0, 1.0]
            "step": self.current_step,
        }

    def set_state(
        self,
        agent_xy: Union[np.ndarray, list, tuple],
        block_xytheta: Union[np.ndarray, list, tuple],
        exact: bool = True,
    ) -> Dict[str, Any]:
        """Teleports agent and T-block to a given state (workspace pixels / radians).

        gym-pusht's _set_state assigns the block position *before* the angle, and pymunk
        rotates bodies about their center of gravity (45 px from the T origin). A single
        call therefore shifts the block origin by up to 90 px depending on the previous
        angle. With exact=True the call is repeated so the old angle already equals the
        new one and the block lands exactly at the requested (x, y, theta).
        exact=False keeps the single-call behaviour (rotation about the center of gravity),
        which is useful for interactive in-place rotation.
        """
        st = [
            float(agent_xy[0]), float(agent_xy[1]),
            float(block_xytheta[0]), float(block_xytheta[1]), float(block_xytheta[2]),
        ]
        unwrapped = self.env.unwrapped
        unwrapped._set_state(st)
        if exact:
            unwrapped._set_state(st)
        raw_obs = unwrapped.get_obs()
        raw_info = unwrapped._get_info()
        self._last_obs = raw_obs
        self._last_info = raw_info
        return self._with_history(self._format_observation(raw_obs, raw_info), None)

    def render(self) -> np.ndarray:
        """Returns rendered RGB frame (96x96x3)."""
        frame = self.env.render()
        if frame is None and self._last_obs is not None:
            frame = self._last_obs["pixels"]
        if frame is not None and frame.shape[:2] != self.render_size:
            frame = cv2.resize(frame, self.render_size, interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        """Closes the environment."""
        self.env.close()


class VideoRecorder:
    """Records environment rollouts with telemetry HUD and exports MP4 video."""

    def __init__(
        self,
        output_path: str,
        fps: int = 15,
        resolution: Tuple[int, int] = (384, 384),
        add_hud: bool = True,
    ):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.add_hud = add_hud
        self.frames = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def record_frame(
        self,
        frame: np.ndarray,
        step: int = 0,
        coverage: float = 0.0,
        fde: Optional[float] = None,
        info_text: str = "",
    ):
        """Appends a frame with telemetry HUD."""
        # Upscale frame for clear visualization
        h, w = self.resolution
        canvas = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.add_hud:
            # Semi-transparent overlay banner at top
            banner_h = 45
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

            # Draw telemetry text
            text_line1 = f"Step: {step:03d} | Coverage: {coverage * 100:.1f}%"
            if fde is not None:
                text_line1 += f" | FDE: {fde:.3f}"
            cv2.putText(
                canvas,
                text_line1,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

            status = "Target Reached (Success)" if coverage >= 0.95 else ("In Progress" if not info_text else info_text)
            color = (0, 255, 0) if coverage >= 0.95 else (200, 200, 200)
            cv2.putText(
                canvas,
                f"Status: {status}",
                (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        self.frames.append(canvas)

    def save(self):
        """Encodes and writes all recorded frames to MP4."""
        if not self.frames:
            return
        imageio.mimwrite(
            self.output_path,
            self.frames,
            fps=self.fps,
            macro_block_size=1,
            format="ffmpeg",
        )
        print(f"Saved rollout video ({len(self.frames)} frames) to {self.output_path}")


class PushTExpertPolicy:
    """Algorithmic push planner for Push-T navigation."""

    def __init__(self, goal_pose: Tuple[float, float, float] = (256.0, 256.0, np.pi / 4.0)):
        self.goal_pose = np.array(goal_pose, dtype=np.float32)

    def get_action(
        self,
        block_pose: np.ndarray,
        agent_pos: np.ndarray,
    ) -> np.ndarray:
        """Computes expert low-level command [x_cmd, y_cmd] in workspace coordinates [0, 512]."""
        bx, by, btheta = float(block_pose[0]), float(block_pose[1]), float(block_pose[2])
        gx, gy, gtheta = float(self.goal_pose[0]), float(self.goal_pose[1]), float(self.goal_pose[2])

        dtheta = np.arctan2(np.sin(gtheta - btheta), np.cos(gtheta - btheta))
        dx = gx - bx
        dy = gy - by
        dist = np.hypot(dx, dy)

        cos_t, sin_t = np.cos(btheta), np.sin(btheta)

        def l2w(lx: float, ly: float) -> np.ndarray:
            return np.array([bx + cos_t * lx - sin_t * ly, by + sin_t * lx + cos_t * ly], dtype=np.float32)

        # 1. Orientation adjustment
        if abs(dtheta) > 0.18:
            rot_dir = np.sign(dtheta)
            contact = l2w(rot_dir * 30.0, 100.0)
            push_target = l2w(rot_dir * 70.0, 100.0)
            ag_dist = np.hypot(agent_pos[0] - contact[0], agent_pos[1] - contact[1])
            if ag_dist > 25.0:
                cmd = contact
            else:
                cmd = push_target
        # 2. Position translation
        else:
            if dist > 6.0:
                ux, uy = dx / dist, dy / dist
                com = l2w(0.0, 45.0)
                contact = com - np.array([ux, uy], dtype=np.float32) * 55.0
                ag_dist = np.hypot(agent_pos[0] - contact[0], agent_pos[1] - contact[1])
                if ag_dist > 30.0:
                    cmd = contact
                else:
                    cmd = com + np.array([ux, uy], dtype=np.float32) * 45.0
            else:
                cmd = agent_pos

        return np.clip(cmd, WORKSPACE_MIN + 10.0, WORKSPACE_MAX - 10.0)

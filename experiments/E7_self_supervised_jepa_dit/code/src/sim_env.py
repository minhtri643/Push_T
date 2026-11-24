"""Thin wrapper around gym-pusht (0.1.8): env creation, direct state setting, rollouts.

Action = target pusher position [x, y] in [0, 512]; a PD controller moves the pusher
(10 physics sub-steps of dt=0.01 per control step, i.e. 10 Hz control).
"""
from __future__ import annotations

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np

from src.utils import wrap_angle


def make_env(obs_type: str = "pixels_agent_pos", render_mode: str = "rgb_array") -> gym.Env:
    """obs_type: 'pixels_agent_pos' (policy rollouts) or 'state' (fast, no rendering)."""
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode)


def set_env_state(env, state5, order: str = "angle_first"):
    """Set [agent_x, agent_y, block_x, block_y, block_angle] directly on pymunk bodies.

    Must be called AFTER env.reset(). Returns a fresh observation (same type as the env's).
    'angle_first' keeps block.position exactly equal to the given value (setting the angle
    after the position rotates around the centre of gravity and shifts the body origin).
    """
    u = env.unwrapped
    s = np.asarray(state5, dtype=np.float64)
    u.agent.position = (float(s[0]), float(s[1]))
    if order == "angle_first":
        u.block.angle = float(s[4])
        u.block.position = (float(s[2]), float(s[3]))
    elif order == "position_first":  # what gym-pusht's own reset_to_state does
        u.block.position = (float(s[2]), float(s[3]))
        u.block.angle = float(s[4])
    else:
        raise ValueError(f"Unknown order: {order}")
    u.agent.velocity = (0.0, 0.0)
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)
    return u.get_obs()


def get_block_pose(env) -> np.ndarray:
    """[x, y, angle] of the T block, angle wrapped to [-pi, pi]."""
    u = env.unwrapped
    return np.array([u.block.position[0], u.block.position[1], wrap_angle(u.block.angle)], dtype=np.float64)


def get_agent_pos(env) -> np.ndarray:
    u = env.unwrapped
    return np.array([u.agent.position[0], u.agent.position[1]], dtype=np.float64)


def get_coverage(env) -> float:
    return float(env.unwrapped._get_coverage())


def rollout_targets(env, targets: np.ndarray, stop_on_success: bool = False, record: bool = False) -> dict:
    """Execute target positions [H, 2] (pixel space 0..512), one per control step (0.1 s).

    Uses env.unwrapped.step to bypass TimeLimit / termination wrappers.
    Returns dict: block_poses [h, 3], agent_pos [h, 2], coverage [h], success_step (1-based or None),
    and frames (list of RGB images) if record=True. h <= H (shorter when stop_on_success).
    """
    u = env.unwrapped
    block_poses, agent_pos, coverage = [], [], []
    frames: list[np.ndarray] = []
    success_step = None
    for i, tgt in enumerate(np.asarray(targets, dtype=np.float64)):
        _, _, _, _, info = u.step(tgt)
        block_poses.append(get_block_pose(env))
        agent_pos.append(get_agent_pos(env))
        coverage.append(float(info["coverage"]))
        if record:
            frames.append(env.render())
        if success_step is None and bool(info["is_success"]):
            success_step = i + 1
            if stop_on_success:
                break
    out = {
        "block_poses": np.array(block_poses),
        "agent_pos": np.array(agent_pos),
        "coverage": np.array(coverage),
        "success_step": success_step,
    }
    if record:
        out["frames"] = frames
    return out


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset(seed=0)
    print("obs:", {k: (v.shape, v.dtype) for k, v in obs.items()})
    assert obs["pixels"].shape == (96, 96, 3) and obs["pixels"].dtype == np.uint8

    rng = np.random.RandomState(0)
    for _ in range(5):
        _, r, term, trunc, info = env.unwrapped.step(rng.uniform(0, 512, size=2))
    print("info keys after step:", sorted(info.keys()))
    print("block pose:", get_block_pose(env), "| coverage:", get_coverage(env))

    # set_env_state: which order reproduces the requested block position?
    target_state = np.array([256.0, 256.0, 300.0, 200.0, 1.0])
    for order in ("angle_first", "position_first"):
        env.reset(seed=1)
        obs = set_env_state(env, target_state, order)
        pose = get_block_pose(env)
        pos_err = np.linalg.norm(pose[:2] - target_state[2:4])
        ang_err = np.degrees(abs(wrap_angle(pose[2] - target_state[4])))
        print(f"{order:15s} block pos error = {pos_err:7.3f} px | angle error = {ang_err:6.3f} deg")
        assert not np.isnan(obs["pixels"]).any()

    # rollout: holding the pusher still must not move it
    env.reset(seed=2)
    res = rollout_targets(env, np.tile(get_agent_pos(env), (20, 1)), record=True)
    print("rollout shapes:", res["block_poses"].shape, res["agent_pos"].shape, res["coverage"].shape,
          "| frame:", res["frames"][0].shape, "| success_step:", res["success_step"])
    assert len(res["frames"]) == 20 and res["block_poses"].shape == (20, 3)
    env.close()
    print("sim_env dummy test passed")

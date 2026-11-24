"""Inspect the installed gym-pusht: attributes, spaces, info keys and key method sources.

Run: python -m scripts.inspect_sim
"""
from __future__ import annotations

import inspect
from importlib.metadata import version

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np

KEYWORDS = ("reset", "step", "state", "obs", "info", "coverage", "reward", "success")


def main() -> None:
    print("gym-pusht", version("gym-pusht"), "| gymnasium", version("gymnasium"), "| pymunk", version("pymunk"))

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    print("\nenv:", env)
    print("max_episode_steps:", env.spec.max_episode_steps)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    u = env.unwrapped
    print("\nclass:", type(u).__module__ + "." + type(u).__name__)

    print("\n--- scalar / small attributes ---")
    for name in ["dt", "control_hz", "k_p", "k_v", "legacy", "success_threshold", "window_size",
                 "render_size", "goal_pose", "observation_width", "observation_height"]:
        if hasattr(u, name):
            print(f"{name} = {getattr(u, name)!r}")

    print("\n--- all public attributes ---")
    print([n for n in dir(u) if not n.startswith("__")])

    obs, info = env.reset(seed=0)
    print("\n--- reset(seed=0) ---")
    print({k: (v.shape, v.dtype) for k, v in obs.items()})
    print("info keys:", {k: type(v).__name__ for k, v in info.items()})

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("\n--- step(random) ---")
    print("reward:", reward, "| terminated:", terminated, "| truncated:", truncated)
    print("info keys:", {k: type(v).__name__ for k, v in info.items()})
    print("block position/angle:", np.array(list(u.block.position) + [u.block.angle]))

    print("\n--- source of relevant methods ---")
    seen = set()
    for name in dir(type(u)):
        if name.startswith("__") or not any(k in name.lower() for k in KEYWORDS):
            continue
        fn = getattr(type(u), name, None)
        if not callable(fn) or name in seen:
            continue
        seen.add(name)
        try:
            print(f"\n##### {name}")
            print(inspect.getsource(fn))
        except (OSError, TypeError):
            print("(source not available)")


if __name__ == "__main__":
    main()

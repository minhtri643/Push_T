"""Diagnose why expert demos never exceed 0.95 coverage in gym-pusht.

Run: python -m scripts.diag_coverage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import episode_bounds, load_cache, read_lerobot_table
from src.sim_env import get_coverage, make_env, rollout_targets, set_env_state
from src.utils import load_config

FIG_DIR = Path("outputs/figures/step2")


def main() -> None:
    cfg = load_config("configs/default.yaml")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cfg["data"]["cache_dir"])
    state5, action = np.asarray(cache["state5"]), np.asarray(cache["action"])
    bounds = episode_bounds(cache["episode_ends"])
    reward = read_lerobot_table(Path(cfg["data"]["lerobot_dir"]))["next.reward"].to_numpy(dtype=np.float64)

    env = make_env("state")
    env.reset(seed=0)
    u = env.unwrapped
    print("goal_pose:", np.array(u.goal_pose), "| block_cog:", u.block_cog, "| damping:", u.damping,
          "| success_threshold:", u.success_threshold)

    print("\n=== 1. Coverage when the T sits exactly on the goal pose ===")
    gx, gy, gth = (float(v) for v in u.goal_pose)
    for order in ("angle_first", "position_first"):
        env.reset(seed=0)
        set_env_state(env, [100.0, 100.0, gx, gy, gth], order)
        print(f"{order:15s}: coverage = {get_coverage(env):.4f}")

    print("\n=== 2. Replay all episodes; compare sim coverage with dataset next.reward ===")
    rows = []
    for ep, (s, e) in enumerate(bounds):
        env.reset(seed=0)
        set_env_state(env, state5[s], "angle_first")
        cov = rollout_targets(env, action[s:e].astype(np.float64))["coverage"]  # cov[k]: after k+1 actions
        rew = reward[s:e]
        rows.append({
            "episode": ep, "len": e - s,
            "max_cov": cov.max(), "final_cov": cov[-1],
            "max_rew": rew.max(), "final_rew": rew[-1],
            "err_same": np.abs(cov - rew).mean(),
            "err_scaled": np.abs(cov / 0.95 - rew).mean(),
            "err_rew_lead": np.abs(cov[:-1] - rew[1:]).mean(),
            "err_rew_lag": np.abs(cov[1:] - rew[:-1]).mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "diag_coverage.csv", index=False)

    print(f"episodes: {len(df)}")
    print(f"max_cov  percentiles 0/25/50/75/100: {np.percentile(df['max_cov'], [0, 25, 50, 75, 100]).round(3)}")
    print(f"final_cov percentiles 0/25/50/75/100: {np.percentile(df['final_cov'], [0, 25, 50, 75, 100]).round(3)}")
    print(f"episodes with max_cov > 0.95: {(df['max_cov'] > 0.95).sum()} / {len(df)}")
    print(f"episodes with max_cov > 0.90: {(df['max_cov'] > 0.90).sum()} / {len(df)}")
    print(f"dataset reward: max over episodes={df['max_rew'].max():.3f}, mean final={df['final_rew'].mean():.3f}")
    print("mean |sim_coverage - dataset_reward| for 4 alignments:")
    for c in ["err_same", "err_scaled", "err_rew_lead", "err_rew_lag"]:
        print(f"  {c:13s}: {df[c].mean():.5f}")
    print("\nsaved:", FIG_DIR / "diag_coverage.csv")


if __name__ == "__main__":
    main()

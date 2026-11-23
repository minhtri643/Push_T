"""Stage E5a (legacy): choose the hand-over threshold of the vision-only policy on DEVELOPMENT seeds.

dit_vwm lets the vision World Model choose actions while the block is far from the goal and hands
over to the inverse dynamics once the coverage estimated from the image exceeds `gate`.
The gate is chosen here on seeds 90000+ (never used by scripts/5_evaluation/benchmark.py), then frozen in
the config; the benchmark is run once afterwards.

    python experiments/E5_vision_world_models/code_legacy_dinov2/tune_vision_gate.py --gates none 0.5 0.7 0.85 --episodes 30
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json
import time

import numpy as np
import torch
import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.policy import build_policy, run_episode


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", nargs="+", default=["none", "0.5", "0.7", "0.85"])
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    policy = build_policy(cfg, "cuda", kind="dit_vwm")
    sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=(96, 96), max_steps=cfg["env"]["max_steps"])
    out = {}
    for g in args.gates:
        policy.vwm_gate_coverage = None if g == "none" else float(g)
        res, t0 = [], time.perf_counter()
        for j in range(args.episodes):
            torch.manual_seed(j)
            res.append(run_episode(sim, policy, seed=90_000 + j, max_steps=cfg["env"]["max_steps"], exec_horizon=cfg["evaluation"]["exec_horizon"]))
        cov = float(np.mean([r["max_coverage"] for r in res]))
        succ = float(np.mean([r["success"] for r in res]))
        out[g] = {"success": succ, "mean_max_coverage": cov}
        print(f"gate {g:>5}: success {succ * 100:5.1f}% | mean max coverage {cov * 100:5.1f}% | {time.perf_counter() - t0:.0f}s", flush=True)
    os.makedirs("results/vision_world_models/E5a_dinov2_zarr_sim", exist_ok=True)
    with open("results/vision_world_models/E5a_dinov2_zarr_sim/gate_tuning_dev_seeds.json", "w") as f:
        json.dump({"dev_seeds": f"90000-{90000 + args.episodes - 1}", "results": out}, f, indent=2)
    best = max(out, key=lambda k: (out[k]["success"], out[k]["mean_max_coverage"]))
    print(f"Best on development seeds: gate {best}")


if __name__ == "__main__":
    main()

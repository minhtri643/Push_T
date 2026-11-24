"""Closed-loop evaluation for Stage E6a:
End-to-End ResNet-18 + Spatial-Softmax + DiT Action Diffusion Policy.

Evaluates on the standard 95 starts (seeds50, hard20, random25) using max 300 steps, success = coverage > 0.95.
Vision-only, no Inverse Dynamics, no privileged block pose.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.utils.normalizer import unnormalize_pos
from experiments.E6_end2end_vision_diffusion.code_E6a_resnet_spatial_direct.model import ResNetSpatialDirectPolicy

P = math.pi
HARD20 = [  # (name, block x, y, theta, pusher x, y)
    ("Center-Bottom", 256, 340, P / 4, 256, 420), ("Center-Right", 340, 256, P / 4, 420, 256),
    ("Center-Top", 256, 170, P / 4, 256, 100), ("Center-Left", 170, 256, P / 4, 100, 256),
    ("Q1 normal angle", 360, 160, 0.5, 400, 120), ("Q1 inverted angle", 360, 160, -P / 2, 320, 200),
    ("Q1 near top wall", 320, 90, 0.0, 320, 40), ("Q2 standard", 150, 160, P / 4, 100, 120),
    ("Q2 opposite angle", 150, 160, -2.0, 180, 200), ("Q2 near left wall", 90, 200, 1.0, 40, 200),
    ("Q3 standard", 160, 360, P / 4, 120, 420), ("Q3 extreme angle", 160, 360, P, 200, 320),
    ("Q3 near corner", 100, 420, -0.5, 60, 450), ("Q4 standard", 360, 360, P / 4, 420, 420),
    ("Q4 rotated 90", 360, 360, -P / 4, 320, 400), ("Q4 near right wall", 430, 320, 0.0, 470, 320),
    ("Pusher opposite side", 180, 350, 0.78, 400, 120), ("Pusher behind goal", 280, 300, 0.78, 200, 200),
    ("Inverted at center", 256, 256, -P * 0.75, 320, 320), ("Tilted 180 at center", 256, 256, P + 0.78, 180, 180),
]


def starts() -> List[Tuple[str, str, Dict[str, object]]]:
    out = [("seeds50", f"seed {42 + 101 * i}", {"seed": 42 + 101 * i}) for i in range(50)]
    out += [("hard20", n, {"seed": 1000 + i, "init_state": [ax, ay, bx, by, th]}) for i, (n, bx, by, th, ax, ay) in enumerate(HARD20)]
    rng = np.random.RandomState(2026)
    for i in range(25):
        bx, by, bth = rng.uniform(110, 400), rng.uniform(110, 400), rng.uniform(-P, P)
        ax, ay = rng.uniform(90, 420), rng.uniform(90, 420)
        out.append(("random25", f"user {i + 1}", {"seed": 2000 + i, "init_state": [ax, ay, bx, by, bth]}))
    return out


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    covs = [r["max_coverage"] for r in rows]
    succs = [r["success"] for r in rows]
    steps = [r["steps_to_success"] for r in rows if r["success"]]
    return {
        "n": len(rows),
        "mean_max_coverage": float(np.mean(covs)),
        "success_rate": float(np.mean(succs)),
        "gte_80": float(np.mean([c >= 0.80 for c in covs])),
        "lt_50": float(np.mean([c < 0.50 for c in covs])),
        "median_steps": float(np.median(steps)) if steps else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/E6_vision/resnet_spatial_direct.pt", help="checkpoint path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=300, help="max steps per episode")
    p.add_argument("--sample-steps", type=int, default=10, help="diffusion DDIM sample steps")
    p.add_argument("--only", nargs="+", help="run only specific suite(s), e.g. seeds50 hard20")
    args = p.parse_args()

    out_dir = "results/vision_world_models/E6a_resnet_spatial_direct"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}. Run train.py first.")

    print(f"Loading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    model = ResNetSpatialDirectPolicy(
        num_keypoints=ckpt.get("num_kp", 16),
        horizon=ckpt.get("horizon", 16),
        model_dim=256,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4.0,
        diffusion_steps=100,
        beta_schedule="squaredcos_cap_v2",
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 1, 3, 1, 1)

    sim = PushTSim(env_name="gym_pusht/PushT-v0", render_size=(96, 96), max_steps=args.max_steps)
    all_starts = starts()
    if args.only:
        all_starts = [s for s in all_starts if s[0] in set(args.only)]

    print(f"Evaluating E6a on {len(all_starts)} starts (max {args.max_steps} steps, success > 0.95)...")
    rows = []
    t0 = time.perf_counter()

    for idx, (suite, name, init) in enumerate(all_starts):
        torch.manual_seed(10_000 + idx)
        np.random.seed(10_000 + idx)

        seed = init.get("seed")
        init_state = init.get("init_state")
        if init_state is not None:
            sim.reset(seed=seed)
            obs = sim.set_state(init_state[:2], init_state[2:5])
        else:
            obs, _ = sim.reset(seed=seed)

        done = False
        step = 0
        max_cov = float(obs["coverage"])
        succ_step = None
        action_queue = []

        while step < args.max_steps and not done:
            if len(action_queue) == 0:
                with torch.no_grad():
                    # Observation history: 2 frames (prev, curr)
                    f_prev = obs.get("prev_pixels", obs["pixels"])
                    f_curr = obs["pixels"]
                    raw_clip = np.stack([f_prev, f_curr], axis=0).astype(np.float32) / 255.0  # (2, 96, 96, 3)
                    clip_t = torch.from_numpy(raw_clip).permute(0, 3, 1, 2).unsqueeze(0).to(args.device)  # (1, 2, 3, 96, 96)
                    clip_norm = (clip_t - mean) / std

                    # Agent positions: (prev, curr) in [-1, 1]
                    p_prev = np.asarray(obs.get("prev_agent_pos", obs["agent_pos"]), dtype=np.float32)
                    p_curr = np.asarray(obs["agent_pos"], dtype=np.float32)
                    agents_t = torch.from_numpy(np.stack([p_prev, p_curr], axis=0)).unsqueeze(0).to(args.device)  # (1, 2, 2)

                    pred_acts = model.sample_actions(clip_norm, agents_t, num_inference_steps=args.sample_steps)[0].cpu().numpy()
                    # Execute first 8 actions (Action Chunking = 8)
                    action_queue = [unnormalize_pos(a) for a in pred_acts[:8]]

            act = action_queue.pop(0)
            obs, _, term, _, _ = sim.step(act, normalized_action=False)
            step += 1
            max_cov = max(max_cov, float(obs["coverage"]))

            if obs["coverage"] > 0.95 and succ_step is None:
                succ_step = step

            if term or (succ_step is not None and step >= succ_step + 10):
                done = True

        is_success = bool(max_cov > 0.95)
        rows.append({
            "suite": suite,
            "name": name,
            "max_coverage": max_cov,
            "success": is_success,
            "steps_to_success": succ_step,
            "total_steps": step,
        })
        status_sym = "[OK]" if is_success else " -- "
        print(f"  [{idx + 1:02d}/{len(all_starts)}] {suite}/{name:<20}: {status_sym} Max: {max_cov * 100:5.1f}% | Steps: {step:3d}")

    sim.close()
    elapsed = time.perf_counter() - t0

    overall = summarize(rows)
    by_suite = {s: summarize([r for r in rows if r["suite"] == s]) for s in ("seeds50", "hard20", "random25") if any(r["suite"] == s for r in rows)}

    print("\n" + "=" * 80)
    print("STAGE E6a EVALUATION SUMMARY:")
    print("=" * 80)
    print(f"Total starts: {overall['n']} | Elapsed: {elapsed:.1f}s")
    print(f"Mean Max Coverage: {overall['mean_max_coverage'] * 100:.1f}%")
    print(f"Success Rate (>95%): {overall['success_rate'] * 100:.1f}%")
    print(f"Coverage >= 80%: {overall['gte_80'] * 100:.1f}%")
    print(f"Severe Failure (<50%): {overall['lt_50'] * 100:.1f}%")
    if overall["median_steps"]:
        print(f"Median steps to success: {overall['median_steps']:.0f}")

    for s, res in by_suite.items():
        print(f"  Suite {s:<10}: Coverage {res['mean_max_coverage'] * 100:.1f}% | Success {res['success_rate'] * 100:.1f}%")

    out_json = {
        "stage": "E6a",
        "description": "End-to-End ResNet-18 + Spatial-Softmax + DiT Action Diffusion (No Inverse Dynamics, T_obs=2)",
        "overall": overall,
        "by_suite": by_suite,
        "episodes": rows,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)


if __name__ == "__main__":
    main()

"""E6a: writes the evaluation of code_E6a_resnet_spatial_direct/eval.py in the common benchmark format.

    python experiments/E6_end2end_vision_diffusion/export_benchmark_json.py

Input : results/vision_world_models/E6a_resnet_spatial_direct/eval_metrics.json (written by eval.py)
Output: results/benchmark/E6a_resnet_spatial_direct.json (same layout as scripts/5_evaluation/benchmark.py), so that
        scripts/5_evaluation/summarize_results.py puts E6a in every results table.

eval.py uses the same 95 starts as scripts/5_evaluation/benchmark.py (identical HARD20 list and random25 generator,
checked) and the same per-start torch seed 10_000 + k. Its episode loop sends the predicted actions without the
40 px/step limit of pusht_wm_dit.policy.run_episode, and stops 10 steps after the first success (this does not change
max coverage > 0.95).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(ROOT, "results/vision_world_models/E6a_resnet_spatial_direct/eval_metrics.json")
DST = os.path.join(ROOT, "results/benchmark/E6a_resnet_spatial_direct.json")
SUITES = ("seeds50", "hard20", "random25")


def summary(eps):
    mc = np.array([e["max_coverage"] for e in eps])
    steps = [e["steps_to_success"] for e in eps if e["success"]]
    return {"num_episodes": len(eps), "mean_max_coverage": float(mc.mean()), "success_rate": float((mc > 0.95).mean()),
            "rate_max_cov_ge_0.80": float((mc >= 0.80).mean()), "rate_max_cov_lt_0.50": float((mc < 0.50).mean()),
            "median_steps_to_success": float(np.median(steps)) if steps else None}


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    eps = src["episodes"]
    if len(eps) != 95 or [e["suite"] for e in eps].count("seeds50") != 50:
        sys.exit("expected the 95 standard starts")
    out = {"label": "E6a_resnet_spatial_direct", "stage": "E6", "policy": "resnet_spatial_direct",
           "ckpt_dir": "checkpoints/E6_vision", "diffusion_noise_offset": 0, "max_steps": 300, "rate_limit_40px": False,
           "source": "results/vision_world_models/E6a_resnet_spatial_direct/eval_metrics.json "
                     "(experiments/E6_end2end_vision_diffusion/code_E6a_resnet_spatial_direct/eval.py)",
           "seconds": src.get("elapsed_seconds"),
           "summary": {s: summary([e for e in eps if e["suite"] == s]) for s in SUITES},
           "overall": summary(eps), "episodes": eps}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    o = out["overall"]
    print(f"{out['label']}: success {o['success_rate'] * 100:.1f}% | coverage {o['mean_max_coverage'] * 100:.1f}% -> {DST}")


if __name__ == "__main__":
    main()

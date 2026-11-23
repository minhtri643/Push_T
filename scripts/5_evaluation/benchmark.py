"""Pipeline 5 (evaluation): paired closed-loop benchmark of policies (replaces comprehensive_evaluation / random_user_benchmark).

Every policy runs from the SAME 95 starts with the same torch seed per start:
  - seeds50  : 50 gym-pusht resets (seed 42 + 101 i)
  - hard20   : 20 hand-picked placements (walls, corners, inverted block, pusher on the wrong side)
  - random25 : 25 random "user" placements (np.random.RandomState(2026))
Protocol: 300 steps, success = coverage > 0.95, no heuristics.

    python scripts/5_evaluation/benchmark.py --policy dit_id --label v4_dit_id
    python scripts/5_evaluation/benchmark.py --policy dit_wm --label v4_dit_wm
    python scripts/5_evaluation/benchmark.py --summarize            # table of every results/benchmark/*.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import glob
import json
import math
import time

import numpy as np
import torch
import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.policy import POLICY_KINDS, build_policy, resolve_device, run_episode, summarize

OUT_DIR = "results/benchmark"
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


def starts():
    out = [("seeds50", f"seed {42 + 101 * i}", {"seed": 42 + 101 * i}) for i in range(50)]
    out += [("hard20", n, {"seed": 1000 + i, "init_state": [ax, ay, bx, by, th]}) for i, (n, bx, by, th, ax, ay) in enumerate(HARD20)]
    rng = np.random.RandomState(2026)
    for i in range(25):
        bx, by, bth = rng.uniform(110, 400), rng.uniform(110, 400), rng.uniform(-P, P)
        ax, ay = rng.uniform(90, 420), rng.uniform(90, 420)
        out.append(("random25", f"user {i + 1}", {"seed": 2000 + i, "init_state": [ax, ay, bx, by, bth]}))
    return out


def run(args, cfg):
    device = resolve_device(cfg)
    policy = build_policy(cfg, device, ckpt_dir=args.ckpt_dir, kind=args.policy, vision_backend=args.vision_backend)
    sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=tuple(cfg["env"]["render_size"]), max_steps=args.max_steps)
    rows, t0 = [], time.perf_counter()
    for k, (suite, name, init) in enumerate(starts()):
        if args.only and suite not in args.only:
            continue
        torch.manual_seed(10_000 + k + 1000 * args.seed_offset)  # same starts, different diffusion noise per offset
        np.random.seed(10_000 + k + 1000 * args.seed_offset)
        r = run_episode(sim, policy, max_steps=args.max_steps, exec_horizon=cfg["evaluation"]["exec_horizon"], **init)
        r.pop("coverage_history")
        r.update({"suite": suite, "name": name})
        rows.append(r)
    sim.close()
    stage = args.label.split("_")[0] if args.label[:1] == "E" and args.label[1:2].isdigit() else None
    out = {"label": args.label, "stage": stage, "policy": args.policy, "ckpt_dir": args.ckpt_dir or "checkpoints/final",
           "diffusion_noise_offset": args.seed_offset, "max_steps": args.max_steps,
           "vision_backend": getattr(policy, "vision", None) and policy.vision.get("source"),
           "seconds": time.perf_counter() - t0,
           "summary": {s: summarize([r for r in rows if r["suite"] == s]) for s in ("seeds50", "hard20", "random25") if any(r["suite"] == s for r in rows)},
           "overall": summarize(rows), "episodes": rows}
    out_dir = os.path.join(OUT_DIR, "diagnostics") if args.only else OUT_DIR  # partial runs stay out of the summary
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.label}.json"), "w") as f:
        json.dump(out, f, indent=1)
    o = out["overall"]
    print(f"{args.label}: coverage {o['mean_max_coverage'] * 100:.1f}% | success {o['success_rate'] * 100:.0f}% | "
          f">=80% {o['rate_max_cov_ge_0.80'] * 100:.0f}% | <50% {o['rate_max_cov_lt_0.50'] * 100:.0f}% | "
          f"median steps to success {o['median_steps_to_success']:.0f} | {out['seconds']:.0f}s")


def sign_test_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize_all(ref_label):
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.json")))
    res = {json.load(open(f))["label"]: json.load(open(f)) for f in files}
    ref = res.get(ref_label)
    lines = ["| Policy | Độ phủ TB | Thành công | ≥80% | <50% | Bước tới thành công (trung vị) | seeds50 | hard20 | random25 | so với " + (ref_label or "-") + " |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in res.items():
        o = r["overall"]
        cmp = "-"
        if ref and lab != ref_label:
            a = np.array([e["max_coverage"] for e in r["episodes"]])
            b = np.array([e["max_coverage"] for e in ref["episodes"]])
            better, worse = int(np.sum(a - b > 0.10)), int(np.sum(b - a > 0.10))
            cmp = f"{better} tốt hơn / {worse} kém hơn (p={sign_test_p(better, worse):.3f})"
        suites = [f"{r['summary'][s]['mean_max_coverage'] * 100:.0f}% / {r['summary'][s]['success_rate'] * 100:.0f}%" for s in ("seeds50", "hard20", "random25")]
        lines.append(f"| {lab} | {o['mean_max_coverage'] * 100:.1f}% | {o['success_rate'] * 100:.0f}% | {o['rate_max_cov_ge_0.80'] * 100:.0f}% | "
                     f"{o['rate_max_cov_lt_0.50'] * 100:.0f}% | {o['median_steps_to_success']:.0f} | " + " | ".join(suites) + f" | {cmp} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Closed-loop benchmark (95 starts, 300 steps, success > 95%)\n\nCột suite: độ phủ TB / thành công.\n\n" + table + "\n")
    print(table)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=POLICY_KINDS, default="dit_id")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--max-steps", type=int, default=cfg["env"]["max_steps"])
    p.add_argument("--only", nargs="*", default=None, help="run only these suites (seeds50 hard20 random25); results go to results/benchmark/diagnostics")
    p.add_argument("--seed-offset", type=int, default=0, help="re-run with different diffusion noise to measure variance")
    p.add_argument("--vision-backend", default=None, help="vision World Model for dit_id_vision / dit_vwm: jepa_wm | vjepa2_wm | dinov2_hf_wm")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--ref", default=None, help="label used as reference in the summary")
    args = p.parse_args()
    if args.summarize:
        summarize_all(args.ref)
        return
    args.label = args.label or f"{args.policy}{'_' + os.path.basename(args.ckpt_dir.rstrip('/')) if args.ckpt_dir else ''}"
    run(args, cfg)


if __name__ == "__main__":
    main()

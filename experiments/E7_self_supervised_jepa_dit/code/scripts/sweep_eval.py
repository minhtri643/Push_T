"""Sweep (n_action_steps : n_infer) on the same 50 seeds; prints one comparison table.

Run: python -m scripts.sweep_eval --configs 4:16 6:16 12:16
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import pandas as pd

from src.eval_sim import PushTAgent, evaluate, load_models
from src.utils import get_device, load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_run", default="wm_main")
    ap.add_argument("--policy_run", default="dit_main")
    ap.add_argument("--configs", nargs="+", default=["4:16", "6:16", "12:16"], help="n_action_steps:n_infer")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--wm_random_init", action="store_true")
    args = ap.parse_args()

    device = get_device()
    wm, policy, cfg = load_models(args.wm_run, args.policy_run, device, wm_random_init=args.wm_random_init)
    ev = load_config("configs/default.yaml")["eval"]
    root = Path("outputs/runs") / args.policy_run / "eval"

    rows = []
    base = root / "main" / "eval_summary.json"
    if base.exists():
        b = json.loads(base.read_text(encoding="utf-8"))
        rows.append({"n_action_steps": b["n_action_steps"], "n_infer": b["n_infer"], "success_095": b["success_rate"],
                     "success_090": b["success_090_rate"], "mean_max_cov": b["mean_max_coverage"],
                     "median_time_s": (b["time_to_success_s"] or {}).get("median"), "infer_ms": b["mean_infer_ms"],
                     "minutes": None, "note": "baseline (main)"})

    for c in args.configs:
        k, n = (int(v) for v in c.split(":"))
        set_seed(cfg["project"]["seed"])
        agent = PushTAgent(wm, policy, cfg, device, n_action_steps=k, n_infer=n)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            s = evaluate(agent, cfg, root / f"{args.name}_k{k}_s{n}", n_episodes=args.n_episodes,
                         seed_start=ev["seed_start"], n_videos=0, max_steps=ev["max_steps"])
        row = {"n_action_steps": k, "n_infer": n, "success_095": s["success_rate"], "success_090": s["success_090_rate"],
               "mean_max_cov": s["mean_max_coverage"], "median_time_s": (s["time_to_success_s"] or {}).get("median"),
               "infer_ms": s["mean_infer_ms"], "minutes": (time.time() - t0) / 60, "note": ""}
        rows.append(row)
        print(f"done k={k} steps={n}: success@0.95={row['success_095'] * 100:.0f}%  "
              f"success@0.90={row['success_090'] * 100:.0f}%  mean_max_cov={row['mean_max_cov']:.3f}  "
              f"({row['minutes']:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(root / f"{args.name}_table.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== COMPARISON (same 50 seeds; sampling std of success rate is about 6 points) ===")
    print(df.round(3).to_string(index=False))
    best = df.sort_values(["mean_max_cov", "success_095"], ascending=False).iloc[0]
    print(f"\nbest by mean_max_cov: n_action_steps={int(best['n_action_steps'])}, n_infer={int(best['n_infer'])}")


if __name__ == "__main__":
    main()

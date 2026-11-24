"""Object-level FDE: where does the T block end up if the sim executes the predicted pusher path?

Run: python -m scripts.eval_offline_fde --policy_run dit_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import episode_bounds, load_cache
from src.dit_policy import DiffusionPolicy
from src.sim_env import make_env, rollout_targets, set_env_state
from src.utils import get_device, load_config, wrap_angle


def final_pose(env, state5_t, targets, order) -> np.ndarray:
    env.reset(seed=0)
    set_env_state(env, state5_t, order)
    return rollout_targets(env, np.asarray(targets, dtype=np.float64))["block_poses"][-1]


def pose_errors(p_hat: np.ndarray, p_ref: np.ndarray, r_theta: float):
    d_pos = float(np.linalg.norm(p_hat[:2] - p_ref[:2]))
    d_th = float(abs(wrap_angle(p_hat[2] - p_ref[2])))               # rad
    return d_pos, float(np.degrees(d_th)), d_pos + r_theta * d_th


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_run", default="dit_main")
    ap.add_argument("--latents", default="data/cache/latents_wm_main.npy")
    ap.add_argument("--n_draws", type=int, default=5)
    ap.add_argument("--n_infer", type=int, default=None)
    ap.add_argument("--out_root", default="outputs/runs")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(Path(args.out_root) / args.policy_run / "ckpt" / "best.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    H = int(cfg["policy"]["horizon"])
    ws = float(cfg["data"]["workspace_size"])
    n_infer = args.n_infer or int(cfg["policy"]["n_inference_steps"])
    ev = load_config("configs/default.yaml")["eval"]
    stride, r_theta = int(ev["fde_stride"]), float(ev["r_theta_px"])
    order = cfg["sim"]["set_state_order"]

    latents = np.load(args.latents)
    cache = load_cache(cfg["data"]["cache_dir"])
    state5 = np.asarray(cache["state5"], dtype=np.float64)
    action = np.asarray(cache["action"], dtype=np.float32)
    apos = np.asarray(cache["agent_pos"], dtype=np.float32)
    ep_idx = np.asarray(cache["episode_index"])
    bounds = episode_bounds(cache["episode_ends"])

    items = [(ep, t) for ep in cache["split"]["val"] for t in range(int(bounds[ep][0]), int(bounds[ep][1]) - H, stride)]
    t_arr = np.array([t for _, t in items])
    N = len(items)
    print(f"{N} val samples (stride {stride}, horizon {H}) from {len(cache['split']['val'])} episodes")

    pol = DiffusionPolicy(cfg["policy"], latents.shape[-1], ws).to(device)
    pol.load_state_dict(ck["model"])
    pol.eval()
    s = torch.from_numpy(latents[0, t_arr]).float().to(device)
    p = torch.from_numpy(apos[t_arr]).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    draws = []
    for _ in range(args.n_draws):
        draws.append(torch.cat([pol.sample(s[i:i + 512], p[i:i + 512], n_infer, generator=gen)
                                for i in range(0, N, 512)]).cpu().numpy())
    draws = np.stack(draws)                                                    # [D, N, H, 2]

    prev = apos[np.maximum(t_arr - 1, 0)]
    same = (ep_idx[t_arr] == ep_idx[np.maximum(t_arr - 1, 0)])[:, None]
    vel = (apos[t_arr] - prev) * same
    steps = np.arange(1, H + 1, dtype=np.float32)[None, :, None]
    const_vel = np.clip(apos[t_arr][:, None, :] + vel[:, None, :] * steps, 0, ws)
    still = np.repeat(apos[t_arr][:, None, :], H, axis=1)

    env = make_env("state")
    rows, drift = [], []
    for i, (ep, t) in enumerate(tqdm(items, desc="sim rollouts")):
        a_gt = action[t:t + H]
        ref = final_pose(env, state5[t], a_gt, order)
        rec = state5[t + H, 2:5]
        dp, dt_, _ = pose_errors(ref, rec, r_theta)
        drift.append((dp, dt_))

        cand = {f"draw{d}": draws[d, i] for d in range(args.n_draws)}
        cand["still"], cand["const_vel"] = still[i], const_vel[i]
        res = {}
        for name, path in cand.items():
            pose = final_pose(env, state5[t], path, order)
            d_pos, d_th, d_obj = pose_errors(pose, ref, r_theta)
            dist = np.linalg.norm(path - a_gt, axis=1)
            res[name] = {"FDE_pos": d_pos, "FDE_theta_deg": d_th, "FDE_obj": d_obj,
                         "FDE_agent": float(dist[-1]), "ADE_agent": float(dist.mean())}
        draw_keys = [f"draw{d}" for d in range(args.n_draws)]
        min5 = {"minFDE_pos@5": min(res[k]["FDE_pos"] for k in draw_keys),
                "minFDE_obj@5": min(res[k]["FDE_obj"] for k in draw_keys)}
        rows.append({"episode": ep, "t": t, "method": "DiT", **res["draw0"], **min5})
        rows.append({"episode": ep, "t": t, "method": "stand still", **res["still"]})
        rows.append({"episode": ep, "t": t, "method": "constant velocity", **res["const_vel"]})
    env.close()

    df = pd.DataFrame(rows)
    out = Path(args.out_root) / args.policy_run / "fde_obj"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "fde_results.csv", index=False)

    cols = ["FDE_pos", "FDE_theta_deg", "FDE_obj", "FDE_agent", "ADE_agent"]
    summ = {m: {c: {"mean": float(g[c].mean()), "median": float(g[c].median()), "std": float(g[c].std())} for c in cols}
            for m, g in df.groupby("method")}
    dg = df[df["method"] == "DiT"]
    summ["DiT"]["minFDE_pos@5"] = float(dg["minFDE_pos@5"].mean())
    summ["DiT"]["minFDE_obj@5"] = float(dg["minFDE_obj@5"].mean())
    dr = np.array(drift)
    summ["sim_vs_record_drift"] = {"pos_px_mean": float(dr[:, 0].mean()), "pos_px_median": float(np.median(dr[:, 0])),
                                   "theta_deg_mean": float(dr[:, 1].mean())}
    summ["n_samples"] = N
    (out / "fde_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n=== Object-level FDE at 2 s (T-block pose vs sim replay of the expert path) ===")
    print(df.groupby("method")[cols].mean().round(2).to_string())
    print(f"\nDiT minFDE@5: pos {summ['DiT']['minFDE_pos@5']:.1f} px | obj {summ['DiT']['minFDE_obj@5']:.1f}")
    print(f"sim vs record drift (noise floor of this measurement): pos mean {dr[:, 0].mean():.2f} px, "
          f"median {np.median(dr[:, 0]):.2f} px | theta mean {dr[:, 1].mean():.2f} deg")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"DiT": ("-", "black"), "stand still": ("--", "dimgray"), "constant velocity": (":", "black")}
    for ax, col, lab in ((axes[0], "FDE_pos", "T-block position error (px)"),
                         (axes[1], "FDE_theta_deg", "T-block angle error (deg)")):
        hi = float(np.percentile(df[col], 99))
        for m, (ls, c) in styles.items():
            ax.hist(df[df["method"] == m][col], bins=40, range=(0, hi), density=True, histtype="step",
                    ls=ls, color=c, lw=1.8, label=m)
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fde_hist.png", dpi=150)
    plt.close(fig)
    print(f"saved fde_results.csv, fde_summary.json, fde_hist.png in {out}")


if __name__ == "__main__":
    main()

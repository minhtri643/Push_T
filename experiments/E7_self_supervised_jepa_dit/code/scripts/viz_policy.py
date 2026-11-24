"""Visualise the trained DiT policy and compute FDE/ADE against baselines.

Run: python -m scripts.viz_policy --run_name dit_main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import LatentPathDataset, load_cache
from src.dit_policy import DiffusionPolicy, path_to_velocity
from src.utils import get_device, unnormalize_pos

SCALE = 96 / 512


def to_px(xy):
    return np.asarray(xy) * SCALE


@torch.no_grad()
def sample_all(pol, ds, idx, device, n_infer=16, bs=512, seed=0, n_draws=1) -> torch.Tensor:
    """-> [n_draws, N, H, 2] px, EMA model, fixed seed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for _ in range(n_draws):
        outs = []
        for i in range(0, len(idx), bs):
            s, p, _, _ = ds.get(idx[i:i + bs])
            outs.append(pol.sample(s, p, n_infer, generator=gen))
        draws.append(torch.cat(outs))
    return torch.stack(draws)


@torch.no_grad()
def denoise_trace(pol, s, p, n_infer, gen, record_ks=(99, 66, 33)) -> dict:
    x = torch.randn(1, pol.H, 2, device=s.device, generator=gen)
    cond = pol._cond(s, p)
    ts = pol.scheduler.ddim_timesteps(n_infer)
    trace = {}
    for i, k in enumerate(ts):
        if k in record_ks:
            trace[k] = unnormalize_pos(x, pol.ws)[0].cpu().numpy()
        k_prev = ts[i + 1] if i + 1 < len(ts) else -1
        eps = pol.dit(x, torch.full((1,), k, device=s.device, dtype=torch.long), cond)
        x = pol.scheduler.ddim_step(x, eps, k, k_prev)
    trace[0] = unnormalize_pos(x.clamp(-1, 1), pol.ws)[0].cpu().numpy()
    return trace


def metrics(pred, gt) -> dict:
    d = (pred - gt).norm(dim=-1)  # [N, H] px
    return {"ADE_px": float(d.mean()), "FDE_1s_px": float(d[:, 9].mean()), "FDE_2s_px": float(d[:, -1].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="dit_main")
    ap.add_argument("--out_root", default="outputs/runs")
    ap.add_argument("--latents", default="data/cache/latents_wm_main.npy")
    args = ap.parse_args()
    device = get_device()
    run_dir = Path(args.out_root) / args.run_name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    latents = np.load(args.latents)
    ck = torch.load(run_dir / "ckpt" / "best.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    H, dt, ws = int(cfg["policy"]["horizon"]), float(cfg["data"]["dt"]), float(cfg["data"]["workspace_size"])
    pol = DiffusionPolicy(cfg["policy"], latents.shape[-1], ws).to(device)
    pol.load_state_dict(ck["model"])
    pol.eval()
    print(f"loaded {run_dir / 'ckpt' / 'best.pt'} (epoch {ck['epoch']}, val_fde {ck['val_fde']:.1f} px)")

    cache = load_cache(cfg["data"]["cache_dir"])
    ds = LatentPathDataset(cache, latents, cache["split"]["val"], H, 1, device)
    idx = torch.nonzero(ds.full_h).squeeze(1)
    N = len(idx)
    s_all, p_all, gt_all, _ = ds.get(idx)
    t_all = ds.t[idx].cpu().numpy()
    print(f"val samples with full horizon: {N}")

    # ---- FDE / ADE table ----------------------------------------------------------------
    draws = sample_all(pol, ds, idx, device, n_draws=8)           # [8, N, H, 2]
    ep = np.asarray(cache["episode_index"])
    same = torch.from_numpy(ep[t_all] == ep[np.maximum(t_all - 1, 0)]).to(device)
    prev_pos = torch.from_numpy(np.asarray(cache["agent_pos"], dtype=np.float32)[np.maximum(t_all - 1, 0)]).to(device)
    v_step = (p_all - prev_pos) * same[:, None]
    steps = torch.arange(1, H + 1, device=device, dtype=torch.float32)[None, :, None]
    const_vel = (p_all[:, None, :] + v_step[:, None, :] * steps).clamp(0, ws)
    still = p_all[:, None, :].expand(-1, H, -1)

    rows = [
        {"method": "DiT (1 sample)", **metrics(draws[0], gt_all)},
        {"method": "stand still", **metrics(still, gt_all)},
        {"method": "constant velocity", **metrics(const_vel, gt_all)},
    ]
    d8 = (draws - gt_all[None]).norm(dim=-1)                     # [8, N, H]
    best = d8.mean(dim=2).argmin(dim=0)                          # best-of-8 by ADE
    pick = draws[best, torch.arange(N, device=device)]
    rows.append({"method": "DiT (best of 8, oracle)", **metrics(pick, gt_all)})
    df = pd.DataFrame(rows)
    df["FDE_2s_vs_still_%"] = 100 * (df["FDE_2s_px"] / rows[1]["FDE_2s_px"] - 1)
    df.to_csv(run_dir / "fde_baselines.csv", index=False)
    print("\n=== Pusher FDE/ADE on all val samples (px, workspace 0..512) ===")
    print(df.round(2).to_string(index=False))
    fde = rows[0]["FDE_2s_px"]
    cond = fde <= 0.7 * min(rows[1]["FDE_2s_px"], rows[2]["FDE_2s_px"])
    print(f"\n[{'PASS' if cond else 'FAIL'}] DiT FDE is at least 30% lower than both baselines")

    # ---- samples_overlay.png ------------------------------------------------------------
    length = (gt_all[:, 1:] - gt_all[:, :-1]).norm(dim=-1).sum(1).cpu().numpy()
    cand = np.where(length > 150)[0]
    rng = np.random.RandomState(0)
    sel = np.sort(rng.choice(cand, size=6, replace=False))
    gen = torch.Generator(device=device).manual_seed(1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, j in zip(axes.ravel(), sel):
        s = s_all[j:j + 1].repeat(8, 1)
        p = p_all[j:j + 1].repeat(8, 1)
        with torch.no_grad():
            smp = pol.sample(s, p, 16, generator=gen).cpu().numpy()
        ax.imshow(np.asarray(cache["images"][t_all[j]]), extent=(0, 96, 96, 0))
        gt = to_px(gt_all[j].cpu().numpy())
        ax.plot(gt[:, 0], gt[:, 1], "-", c="black", lw=2.6, label="expert path (2 s)")
        for m in range(8):
            q = to_px(smp[m])
            ax.plot(q[:, 0], q[:, 1], "--", c="red", lw=1.0, alpha=0.85, label="DiT samples" if m == 0 else None)
            ax.plot(q[-1, 0], q[-1, 1], "x", c="red", ms=7)
        a = to_px(p_all[j].cpu().numpy())
        ax.plot(a[0], a[1], "o", c="white", mec="black", ms=7)
        ax.set_title(f"val frame t={t_all[j]}")
        ax.set_xlim(0, 96)
        ax.set_ylim(96, 0)
        ax.axis("off")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Expert path (solid) vs 8 DiT samples (dashed), X = end point")
    fig.tight_layout()
    fig.savefig(fig_dir / "samples_overlay.png", dpi=140)
    plt.close(fig)

    # ---- denoising_steps.png ------------------------------------------------------------
    j = int(sel[0])
    trace = denoise_trace(pol, s_all[j:j + 1], p_all[j:j + 1], 16, torch.Generator(device=device).manual_seed(2))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, k in zip(axes, (99, 66, 33, 0)):
        ax.imshow(np.asarray(cache["images"][t_all[j]]), extent=(0, 96, 96, 0))
        gt = to_px(gt_all[j].cpu().numpy())
        ax.plot(gt[:, 0], gt[:, 1], "-", c="black", lw=1.5, alpha=0.7)
        q = to_px(trace[k])
        ax.plot(q[:, 0], q[:, 1], "--o", c="red", lw=1.0, ms=3)
        ax.set_title(f"k = {k}" + (" (pure noise)" if k == 99 else " (output)" if k == 0 else ""))
        ax.set_xlim(0, 96)
        ax.set_ylim(96, 0)
        ax.axis("off")
    fig.suptitle("DDIM denoising of one path (black = expert)")
    fig.tight_layout()
    fig.savefig(fig_dir / "denoising_steps.png", dpi=140)
    plt.close(fig)

    # ---- velocity_hist.png --------------------------------------------------------------
    v_hat = path_to_velocity(draws[0], p_all, dt).norm(dim=-1).flatten().cpu().numpy()
    v_gt = path_to_velocity(gt_all, p_all, dt).norm(dim=-1).flatten().cpu().numpy()
    hi = float(np.percentile(np.concatenate([v_hat, v_gt]), 99.5))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(v_gt, bins=50, range=(0, hi), density=True, histtype="step", color="black", lw=1.8, label="expert")
    ax.hist(v_hat, bins=50, range=(0, hi), density=True, histtype="step", color="gray", lw=1.8, ls="--", label="DiT")
    ax.set_xlabel("|v| of the pusher path (px/s)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "velocity_hist.png", dpi=150)
    plt.close(fig)
    print(f"\n|v| mean: expert {v_gt.mean():.1f} px/s | DiT {v_hat.mean():.1f} px/s")
    print(f"saved: samples_overlay.png, denoising_steps.png, velocity_hist.png, fde_baselines.csv in {run_dir}")


if __name__ == "__main__":
    main()

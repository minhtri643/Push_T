"""Evaluate the JEPA world model: error vs k, linear probes (JEPA / random-init / pixel-PCA),
physical prediction error, collapse diagnostics, PCA plots.

Run: python -m scripts.eval_world_model --run_name wm_main
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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.train_world_model import plot_curves
from src.dataset import WMClipDataset, episode_bounds, load_cache
from src.utils import get_device, set_seed, wrap_angle
from src.world_model import JEPAWorldModel, effective_rank, preprocess

TARGET_NAMES = ["agent_x", "agent_y", "block_x", "block_y", "cos_theta", "sin_theta"]


# --------------------------------------------------------------------------- feature extraction
@torch.no_grad()
def encode_all(model, images, device, bs: int = 256):
    """Per-frame online z and LN(EMA z) (the training target) for all frames -> CPU tensors [N, 128]."""
    z, zb = [], []
    for i in range(0, len(images), bs):
        x = preprocess(torch.from_numpy(np.ascontiguousarray(images[i:i + bs])).to(device))
        z.append(model.encoder(x).cpu())
        zb.append(F.layer_norm(model.ema_encoder(x), (model.z_dim,)).cpu())
    return torch.cat(z), torch.cat(zb)


@torch.no_grad()
def compute_S(model, z, prev, device, bs: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, len(z), bs):
        idx = np.arange(i, min(i + bs, len(z)))
        pair = torch.stack([z[prev[idx]], z[idx]], dim=1).to(device)
        out.append(model.aggregator(pair).cpu())
    return torch.cat(out).numpy()


def pixel_pca_features(images, train_mask, n: int = 256) -> np.ndarray:
    feats = []
    for i in range(0, len(images), 2048):
        g = np.asarray(images[i:i + 2048], dtype=np.float32).mean(-1)          # gray [c, 96, 96]
        feats.append(g.reshape(len(g), 48, 2, 48, 2).mean((2, 4)).reshape(len(g), -1))
    X = np.concatenate(feats)
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0).fit(X[train_mask])
    return pca.transform(X).astype(np.float32)


def make_targets(state5: np.ndarray) -> np.ndarray:
    th = state5[:, 4]
    return np.stack([state5[:, 0], state5[:, 1], state5[:, 2], state5[:, 3], np.cos(th), np.sin(th)], axis=1)


def angle_err_deg(p, y):
    """p, y [..., 4] = (x, y, cos, sin) -> |wrapped angle difference| in degrees."""
    return np.degrees(np.abs(wrap_angle(np.arctan2(p[..., 3], p[..., 2]) - np.arctan2(y[..., 3], y[..., 2]))))


# --------------------------------------------------------------------------- probes
def fit_probe(X, Y):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, Y)


def probe_row(name, probe, Xva, Yva) -> dict:
    P = probe.predict(Xva)
    row = {"method": name}
    for i, tn in enumerate(TARGET_NAMES):
        row[f"r2_{tn}"] = r2_score(Yva[:, i], P[:, i])
    row["r2_block_pos"] = (row["r2_block_x"] + row["r2_block_y"]) / 2
    row["r2_block_angle"] = (row["r2_cos_theta"] + row["r2_sin_theta"]) / 2
    row["block_pos_err_px"] = float(np.linalg.norm(P[:, 2:4] - Yva[:, 2:4], axis=1).mean())
    row["block_angle_err_deg"] = float(angle_err_deg(P[:, 2:6], Yva[:, 2:6]).mean())
    return row


# --------------------------------------------------------------------------- error vs k
@torch.no_grad()
def eval_over_k(model, ds, device, Pz, Y_all, K) -> dict:
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    s_pred, s_copy, cnt = np.zeros(K), np.zeros(K), np.zeros(K)
    err = {n: {"pos": [], "ang": []} for n in ("wm", "persist", "floor")}
    masks = []
    for batch in loader:
        obs = preprocess(batch["obs"].to(device))
        fut = preprocess(batch["future_obs"].to(device))
        act = batch["actions"].to(device)
        m = batch["future_mask"].numpy()
        S = model.aggregator(model.encode_frames(obs))
        z_hat = model.rollout(S, act)
        z_bar = F.layer_norm(model.encode_frames(fut, target=True), (model.z_dim,))
        z_now = F.layer_norm(model.encode_frames(obs[:, -1:], target=True), (model.z_dim,))
        e_pred = (z_hat - z_bar).abs().mean(-1).cpu().numpy()
        e_copy = (z_now - z_bar).abs().mean(-1).cpu().numpy()
        s_pred += (e_pred * m).sum(0)
        s_copy += (e_copy * m).sum(0)
        cnt += m.sum(0)

        t = batch["index"].numpy()
        idx = np.minimum(t[:, None] + np.arange(1, K + 1), len(Y_all) - 1)
        truth = Y_all[idx][..., 2:6]
        B = len(t)
        wm_p = Pz.predict(z_hat.reshape(-1, model.z_dim).cpu().numpy()).reshape(B, K, 4)
        fl_p = Pz.predict(z_bar.reshape(-1, model.z_dim).cpu().numpy()).reshape(B, K, 4)
        pe_p = np.repeat(Pz.predict(z_now.reshape(-1, model.z_dim).cpu().numpy())[:, None, :], K, axis=1)
        for name, p in (("wm", wm_p), ("persist", pe_p), ("floor", fl_p)):
            err[name]["pos"].append(np.linalg.norm(p[..., :2] - truth[..., :2], axis=-1))
            err[name]["ang"].append(angle_err_deg(p, truth))
        masks.append(m)
    M = np.concatenate(masks).astype(bool)
    out = {"l_pred_k": s_pred / cnt, "copy_k": s_copy / cnt}
    for name in err:
        for kind in ("pos", "ang"):
            A = np.concatenate(err[name][kind])
            out[f"{name}_{kind}_k"] = (A * M).sum(0) / M.sum(0)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="wm_main")
    ap.add_argument("--out_root", default="outputs/runs")
    args = ap.parse_args()
    device = get_device()
    run_dir = Path(args.out_root) / args.run_name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(run_dir / "ckpt" / "best.pt", map_location=device, weights_only=False)
    cfg, wm = ckpt["cfg"], ckpt["cfg"]["wm"]
    K = int(wm["k_future"])
    model = JEPAWorldModel(wm).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    set_seed(123)
    rmodel = JEPAWorldModel(wm).to(device).eval()
    print(f"loaded {run_dir / 'ckpt' / 'best.pt'} (epoch {ckpt['epoch']}, val_l_pred {ckpt['val_l_pred']:.4f})")

    cache = load_cache(cfg["data"]["cache_dir"])
    images, state5 = cache["images"], np.asarray(cache["state5"], dtype=np.float64)
    ep = np.asarray(cache["episode_index"])
    bounds = episode_bounds(cache["episode_ends"])
    N = len(ep)
    train_mask = np.isin(ep, cache["split"]["train"])
    val_mask = ~train_mask
    Y = make_targets(state5)
    prev = np.arange(N) - 1
    for s, _ in bounds:
        prev[s] = s

    print("encoding all frames (JEPA, random-init, pixel PCA)...")
    z_j, zb_j = encode_all(model, images, device)
    z_r, _ = encode_all(rmodel, images, device)
    S_j, S_r = compute_S(model, z_j, prev, device), compute_S(rmodel, z_r, prev, device)
    X_pix = pixel_pca_features(images, train_mask)

    # ---- linear probes ---------------------------------------------------------------
    rows = []
    for name, X in (("JEPA S_t", S_j), ("random-init S_t", S_r), ("pixel PCA-256", X_pix)):
        probe = fit_probe(X[train_mask], Y[train_mask])
        rows.append(probe_row(name, probe, X[val_mask], Y[val_mask]))
    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(fig_dir / "probe_table.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== Linear probe (Ridge alpha=1, fit on train episodes, scored on val episodes) ===")
    print(probe_df.round(3).to_string(index=False))

    # ---- error vs k (latent + physical) ----------------------------------------------
    Pz = fit_probe(zb_j.numpy()[train_mask], Y[train_mask][:, 2:6])
    val_ds = WMClipDataset(cache, cache["split"]["val"], wm["t_ctx"], K, wm["clip_stride"], in_memory=False)
    res = eval_over_k(model, val_ds, device, Pz, Y, K)
    ks = np.arange(1, K + 1)
    print("\n=== Error vs k (val) ===")
    print("k          :", ks.tolist())
    print("l_pred(k)  :", res["l_pred_k"].round(3).tolist())
    print("copy(k)    :", res["copy_k"].round(3).tolist())
    print("WM pos err (px)   :", res["wm_pos_k"].round(1).tolist())
    print("persist pos (px)  :", res["persist_pos_k"].round(1).tolist())
    print("probe floor pos   :", res["floor_pos_k"].round(1).tolist())
    print("WM angle err (deg):", res["wm_ang_k"].round(1).tolist())
    print("persist angle     :", res["persist_ang_k"].round(1).tolist())
    print("probe floor angle :", res["floor_ang_k"].round(1).tolist())

    plot_curves(run_dir / "log.csv", fig_dir / "curves.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, res["l_pred_k"], "-o", c="black", label="JEPA world model")
    ax.plot(ks, res["copy_k"], "--s", c="black", label="copy baseline")
    ax.set_xlabel("prediction step k (0.1 s each)")
    ax.set_ylabel("latent L1 error (val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_error_vs_k.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind, unit in ((axes[0], "pos", "T-block position error (px, 0..512)"),
                           (axes[1], "ang", "T-block angle error (deg)")):
        ax.plot(ks, res[f"wm_{kind}_k"], "-o", c="black", label="WM rollout + probe")
        ax.plot(ks, res[f"persist_{kind}_k"], "--s", c="black", label="no dynamics (probe on z_t)")
        ax.plot(ks, res[f"floor_{kind}_k"], ":^", c="gray", label="probe on true z (floor)")
        ax.set_xlabel("prediction step k")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "wm_physical_error_vs_k.png", dpi=150)
    plt.close(fig)

    # ---- collapse diagnostics ---------------------------------------------------------
    Sv_j, Sv_r = S_j[val_mask], S_r[val_mask]
    er_j, er_r = effective_rank(torch.from_numpy(Sv_j)), effective_rank(torch.from_numpy(Sv_r))
    std_j, std_r = Sv_j.std(0), Sv_r.std(0)
    print(f"\n=== Collapse diagnostics (val frames) ===\nJEPA : erank={er_j:.1f}  mean std={std_j.mean():.3f}\nrandom: erank={er_r:.1f}  mean std={std_r.mean():.3f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(std_j, bins=30, histtype="step", color="black", lw=1.8, label=f"JEPA (erank {er_j:.0f})")
    ax.hist(std_r, bins=30, histtype="step", color="black", lw=1.5, ls="--", label=f"random-init (erank {er_r:.0f})")
    ax.set_xlabel("std of each S_t dimension")
    ax.set_ylabel("number of dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "s_std_hist.png", dpi=150)
    plt.close(fig)

    # ---- PCA 2D ----------------------------------------------------------------------
    pca = PCA(n_components=2).fit(Sv_j)
    P2 = pca.transform(Sv_j)
    theta = np.arctan2(Y[val_mask][:, 5], Y[val_mask][:, 4])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc0 = axes[0].scatter(P2[:, 0], P2[:, 1], c=Y[val_mask][:, 2], s=4, cmap="viridis")
    axes[0].set_title("val S_t (PCA 2D), coloured by block_x")
    fig.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(P2[:, 0], P2[:, 1], c=theta, s=4, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("coloured by block angle")
    fig.colorbar(sc1, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_S_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for ep_id, ls in zip(cache["split"]["val"][:3], ["-", "--", ":"]):
        s, e = bounds[ep_id]
        T = pca.transform(S_j[s:e])
        ax.plot(T[:, 0], T[:, 1], ls, c="black", lw=1.3)
        ax.scatter(*T[0], marker="o", c="white", edgecolors="black", zorder=3)
        ax.scatter(*T[-1], marker="x", c="black", zorder=3)
    ax.set_title("S_t trajectories of 3 val episodes (o start, x end)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_S_trajectories.png", dpi=150)
    plt.close(fig)

    # ---- pass / fail summary ----------------------------------------------------------
    j, r = rows[0], rows[1]
    gap = res["copy_k"] - res["l_pred_k"]
    checks = {
        "std_s > 0.3": bool(std_j.mean() > 0.3),
        "erank S_t > 20": bool(er_j > 20),
        "l_pred < copy at every k": bool(np.all(res["l_pred_k"] < res["copy_k"])),
        "gap grows with k": bool(gap[-1] > gap[0]),
        "probe R2 block position >= 0.7": bool(j["r2_block_pos"] >= 0.7),
        "probe R2 cos/sin >= 0.5": bool(j["r2_block_angle"] >= 0.5),
        "JEPA > random-init (block pos R2)": bool(j["r2_block_pos"] > r["r2_block_pos"]),
    }
    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")

    summary = {
        "epoch": int(ckpt["epoch"]), "val_l_pred": float(ckpt["val_l_pred"]),
        "erank_jepa": er_j, "erank_random": er_r, "mean_std_jepa": float(std_j.mean()),
        "probe": rows, "error_vs_k": {k: np.asarray(v).tolist() for k, v in res.items()},
        "checks": checks,
    }
    (run_dir / "wm_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved figures + probe_table.csv in {fig_dir} and {run_dir / 'wm_eval.json'}")


if __name__ == "__main__":
    main()

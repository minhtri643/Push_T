"""Final report figures + README GIFs. Run: python -m scripts.make_figures"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
Path("media").mkdir(exist_ok=True)
RUNS = Path("outputs/runs")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

CONFIGS = [
    ("Main", RUNS / "dit_main/eval/main"),
    ("A2 no\nproprio", RUNS / "dit_no_proprio/eval/a2_k8_s16"),
    ("A1 random\nencoder", RUNS / "dit_random_init/eval/a1_k8_s16"),
    ("A3\nk=4", RUNS / "dit_main/eval/sweep_k4_s16"),
    ("A3\nk=6", RUNS / "dit_main/eval/sweep_k6_s16"),
    ("A3\nk=12", RUNS / "dit_main/eval/sweep_k12_s16"),
    ("A4\nDDIM 4", RUNS / "dit_main/eval/a4_k8_s4"),
    ("A4\nDDIM 8", RUNS / "dit_main/eval/a4_k8_s8"),
]


def load_summary(d: Path):
    p = d / "eval_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _wilson(p, n, z=1.96):
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def _yerr(p, n):
    lo, hi = _wilson(p, n)
    return np.vstack([np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)]) * 100


def fig_ablation() -> None:
    items = [(n, load_summary(d)) for n, d in CONFIGS]
    items = [(n, s) for n, s in items if s]
    x = np.arange(len(items))
    s95 = np.array([s["success_rate"] for _, s in items])
    s90 = np.array([s["success_090_rate"] for _, s in items])
    cov = np.array([s["mean_max_coverage"] for _, s in items])
    n = np.array([s["n_episodes"] for _, s in items])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    w = 0.38
    axes[0].bar(x - w / 2, s95 * 100, w, yerr=_yerr(s95, n), color="dimgray", edgecolor="black", capsize=2, label="success @ 0.95")
    axes[0].bar(x + w / 2, s90 * 100, w, yerr=_yerr(s90, n), color="white", edgecolor="black", hatch="//", capsize=2, label="success @ 0.90")
    axes[0].axhline(65.4, ls=":", c="black", lw=1)
    axes[0].text(2, 66.5, "LeRobot DP 65.4%", ha="center", fontsize=8)
    for i, (a, b) in enumerate(zip(s95, s90)):
        if a == 0 and b == 0:
            axes[0].text(i, 9, "0%", ha="center", fontsize=9)
    axes[0].set_ylabel("success rate (%) over 50 seeds (95% Wilson CI)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].bar(x, cov, 0.55, color="white", edgecolor="black", hatch="..")
    axes[1].axhline(0.898, ls="--", c="black", lw=1)
    axes[1].axhline(0.955, ls=":", c="black", lw=1)
    axes[1].text(len(x) - 0.5, 0.905, "expert replay 0.898", ha="right", fontsize=8)
    axes[1].text(len(x) - 0.5, 0.962, "LeRobot DP 0.955", ha="right", fontsize=8)
    axes[1].set_ylabel("mean max coverage")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([n_ for n_, _ in items], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig08_ablation.png", dpi=200)
    plt.close(fig)


def fig_time_box() -> None:
    data, labels = [], []
    for name, d in CONFIGS:
        p = d / "eval_episodes.csv"
        if p.exists():
            t = pd.read_csv(p)["time_to_success_s"].dropna()
            if len(t) >= 3:
                data.append(t.values)
                labels.append(f"{name}\n(n={len(t)})")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=labels, medianprops={"color": "black"}, boxprops={"color": "black"})
    ax.axhline(12.45, ls="--", c="black", lw=1)
    ax.text(0.55, 12.95, "mean expert demo length 12.45 s", fontsize=8, color="black")
    ax.set_ylabel("time to success (s), successful episodes only")
    fig.tight_layout()
    fig.savefig(OUT / "fig08b_time_to_success.png", dpi=200)
    plt.close(fig)


def fig_fde() -> None:
    p = RUNS / "dit_main/fde_obj/fde_summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    methods = ["DiT", "stand still", "constant velocity"]
    hatches = ["", "//", ".."]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, lab in zip(axes, ["FDE_pos", "FDE_theta_deg", "FDE_obj"], ["FDE position (px)", "FDE angle (deg)", "FDE_obj (px)"]):
        vals = [s[m][key]["mean"] for m in methods]
        errs = [s[m][key]["std"] / np.sqrt(s["n_samples"]) for m in methods]
        ax.bar(range(3), vals, yerr=errs, color="white", edgecolor="black", capsize=3)
        for i, h in enumerate(hatches):
            ax.patches[i].set_hatch(h)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["DiT", "stand\nstill", "const.\nvelocity"], fontsize=9)
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_fde.png", dpi=200)
    plt.close(fig)


def read_frames(path: Path):
    r = imageio.get_reader(str(path))
    frames = [f for f in r]
    r.close()
    return frames


def make_gif(frames, out: Path, every: int, size: int = 340) -> None:
    imgs = [Image.fromarray(f).resize((size, size), Image.BILINEAR) for f in frames[::every]]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(imgs)} frames)")


def fig_sequence_and_gifs() -> None:
    vids = RUNS / "dit_main/eval/smoke/videos"
    ok, bad = vids / "ep100001.mp4", vids / "ep100000.mp4"
    if ok.exists():
        frames = read_frames(ok)
        idx = np.linspace(0, len(frames) - 1, 6).astype(int)
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
        for k, (ax, i) in enumerate(zip(axes.ravel(), idx)):
            ax.imshow(frames[i])
            ax.set_title(f"({'abcdef'[k]})", fontsize=14, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "fig09_success_sequence.png", dpi=200)
        plt.close(fig)
        make_gif(frames, Path("media/demo_success.gif"), every=2)
    if bad.exists():
        make_gif(read_frames(bad), Path("media/demo_failure.gif"), every=4)


def copy_existing() -> None:
    pairs = {
        RUNS / "wm_main/figures/curves.png": "fig02_wm_curves.png",
        RUNS / "wm_main/figures/pred_error_vs_k.png": "fig03a_pred_error_vs_k.png",
        RUNS / "wm_main/figures/wm_physical_error_vs_k.png": "fig03b_wm_physical_error.png",
        RUNS / "wm_main/figures/s_std_hist.png": "fig04_s_std_hist.png",
        RUNS / "wm_main/figures/pca_S_scatter.png": "fig05a_pca_scatter.png",
        RUNS / "wm_main/figures/pca_S_trajectories.png": "fig05b_pca_trajectories.png",
        RUNS / "dit_main/figures/samples_overlay.png": "fig06a_samples_overlay.png",
        RUNS / "dit_main/figures/denoising_steps.png": "fig06b_denoising_steps.png",
        RUNS / "dit_main/figures/velocity_hist.png": "fig06c_velocity_hist.png",
        RUNS / "dit_main/figures/curves.png": "fig06d_dit_curves.png",
        RUNS / "dit_main/fde_obj/fde_hist.png": "fig07b_fde_hist.png",
        Path("outputs/figures/step2/replay_drift.png"): "figS1_replay_drift.png",
        Path("outputs/figures/step2/render_vs_dataset.png"): "figS2_render_vs_dataset.png",
    }
    for src, name in pairs.items():
        if src.exists():
            shutil.copy(src, OUT / name)
        else:
            print("missing:", src)


if __name__ == "__main__":
    fig_ablation()
    fig_time_box()
    fig_fde()
    fig_sequence_and_gifs()
    copy_existing()
    print("\nfigures in", OUT)
    for p in sorted(OUT.iterdir()):
        print("  ", p.name)

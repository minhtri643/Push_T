"""Step 1 figures + numeric sanity checks.

Run: python -m scripts.visualize_data
Output: outputs/figures/step1/*.png, ep0.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataset import episode_bounds, load_cache
from src.utils import load_config

SCALE = 96 / 512  # sim coordinates (0..512) -> image pixels (0..96)


def to_px(xy: np.ndarray, flip: bool = False) -> np.ndarray:
    uv = np.asarray(xy, dtype=np.float64) * SCALE
    if flip:
        uv = uv.copy()
        uv[..., 1] = 96.0 - uv[..., 1]
    return uv


def blue_centroid(img: np.ndarray) -> np.ndarray | None:
    """Centroid (continuous pixel coords) of the royal-blue pusher disk."""
    r, g, b = (img[..., i].astype(np.int32) for i in range(3))
    mask = (b > 180) & (r < 120) & (b - r > 100)
    if mask.sum() < 5:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def check_orientation(images, agent_pos, n: int = 300, seed: int = 0) -> dict:
    """Compare projected agent_pos with the detected pusher centroid (direct vs flipped v)."""
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(images), size=n, replace=False))
    err = {"direct": [], "flip": []}
    for i in idx:
        c = blue_centroid(np.asarray(images[i]))
        if c is None:
            continue
        err["direct"].append(np.linalg.norm(to_px(agent_pos[i]) - c))
        err["flip"].append(np.linalg.norm(to_px(agent_pos[i], flip=True) - c))
    res = {k: float(np.mean(v)) for k, v in err.items() if v}
    res["n_used"] = len(err["direct"])
    res["best"] = "flip" if res.get("flip", 1e9) < res.get("direct", 1e9) else "direct"
    return res


def lag_table(agent_pos, action, bounds, max_k: int = 6) -> dict[int, float]:
    """Mean |agent_pos[t+k] - action[t]| in sim units (0..512), inside episodes."""
    out = {}
    for k in range(max_k + 1):
        errs = []
        for s, e in bounds:
            if e - s <= k:
                continue
            errs.append(np.linalg.norm(agent_pos[s + k : e] - action[s : e - k], axis=1))
        out[k] = float(np.concatenate(errs).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path("outputs/figures/step1")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cfg["data"]["cache_dir"])
    images, agent_pos = cache["images"], np.asarray(cache["agent_pos"])
    action, state5 = np.asarray(cache["action"]), np.asarray(cache["state5"])
    bounds = episode_bounds(cache["episode_ends"])
    lengths = bounds[:, 1] - bounds[:, 0]

    # ---- numeric checks ------------------------------------------------------
    ori = check_orientation(images, agent_pos)
    flip = ori["best"] == "flip"
    print("\n=== Orientation check (agent_pos -> pixel vs detected blue pusher) ===")
    print(f"frames used: {ori['n_used']}")
    print(f"mean pixel error  direct v : {ori.get('direct', float('nan')):.3f} px")
    print(f"mean pixel error  flipped v: {ori.get('flip', float('nan')):.3f} px")
    print(f"=> use {'v = 96 - v (FLIP)' if flip else 'v = y * 96/512 (DIRECT)'}")

    lags = lag_table(agent_pos, action, bounds)
    print("\n=== agent_pos[t+k] vs action[t]  (mean error, sim units) ===")
    for k, v in lags.items():
        print(f"k={k}: {v:7.3f}")
    print(f"=> agent_pos follows action with the smallest error at k={min(lags, key=lags.get)}")

    print("\n=== Episode lengths ===")
    print(f"n={len(lengths)} min={lengths.min()} max={lengths.max()} mean={lengths.mean():.2f} median={np.median(lengths):.0f}")

    # ---- episode 0 -----------------------------------------------------------
    s, e = bounds[0]
    L = e - s
    ep_imgs = np.asarray(images[s:e])
    ep_pos = to_px(agent_pos[s:e], flip)
    ep_blk = to_px(state5[s:e, 2:4], flip)

    # 1) montage
    idx = np.linspace(0, L - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(ep_imgs[i], extent=(0, 96, 96, 0))
        ax.scatter(*ep_pos[i], c="red", s=30, edgecolors="white", linewidths=0.8)
        ax.set_title(f"frame {i}")
        ax.axis("off")
    fig.suptitle("Episode 0: 8 evenly spaced frames (red dot = agent_pos)")
    fig.tight_layout()
    fig.savefig(out_dir / "ep0_montage.png", dpi=150)
    plt.close(fig)

    # 2) trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ep_imgs[0], extent=(0, 96, 96, 0))
    ax.plot(ep_pos[:, 0], ep_pos[:, 1], "-", c="red", lw=1.8, label="pusher (agent_pos)")
    ax.plot(ep_blk[:, 0], ep_blk[:, 1], "--", c="black", lw=1.8, label="T-block position (zarr, eval only)")
    ax.scatter(*ep_pos[0], c="red", s=40, edgecolors="white", zorder=3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Episode 0 trajectories on the first frame")
    ax.set_xlim(0, 96)
    ax.set_ylim(96, 0)
    fig.tight_layout()
    fig.savefig(out_dir / "ep0_trajectory.png", dpi=150)
    plt.close(fig)

    # 3) lengths histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=25, edgecolor="black", color="lightgray")
    ax.axvline(lengths.mean(), c="black", ls="--", label=f"mean = {lengths.mean():.1f}")
    ax.set_xlabel("episode length (frames, 10 fps)")
    ax.set_ylabel("number of episodes")
    ax.set_title(f"Episode lengths ({len(lengths)} episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_lengths.png", dpi=150)
    plt.close(fig)

    # 4) state vs action
    t = np.arange(L) * cfg["data"]["dt"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for j, name in enumerate(["x", "y"]):
        axes[j].plot(t, agent_pos[s:e, j], "-", c="black", label="agent_pos")
        axes[j].plot(t, action[s:e, j], "--", c="red", label="action")
        axes[j].set_ylabel(f"{name} (sim units)")
        axes[j].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Episode 0: agent_pos (solid) vs action (dashed)")
    fig.tight_layout()
    fig.savefig(out_dir / "ep0_state_vs_action.png", dpi=150)
    plt.close(fig)

    # 5) gif (4x nearest upscale -> 384x384)
    big = [Image.fromarray(np.repeat(np.repeat(f, 4, axis=0), 4, axis=1)) for f in ep_imgs]
    big[0].save(out_dir / "ep0.gif", save_all=True, append_images=big[1:], duration=100, loop=0)

    print("\nSaved:")
    for p in sorted(out_dir.iterdir()):
        print("  ", p)


if __name__ == "__main__":
    main()

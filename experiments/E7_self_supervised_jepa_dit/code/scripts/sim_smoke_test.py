"""Step 2 smoke test: env info + speed (A), random agent (B), demo replay (C),
render vs dataset (D), replay video (E).

Run: python -m scripts.sim_smoke_test --config configs/default.yaml --parts A B C D E
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.dataset import episode_bounds, load_cache
from src.sim_env import get_block_pose, get_coverage, make_env, rollout_targets, set_env_state
from src.utils import load_config, set_seed, wrap_angle

FIG_DIR = Path("outputs/figures/step2")
VID_DIR = Path("outputs/videos/step2")


# --------------------------------------------------------------------------- A
def part_a(cfg: dict) -> None:
    print("\n=== A. Env info and speed ===")
    env = make_env()
    obs, info = env.reset(seed=0)
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    print("pixels:", obs["pixels"].shape, obs["pixels"].dtype, "| agent_pos:", obs["agent_pos"].shape)
    print("info keys at reset:", sorted(info.keys()))
    _, _, _, _, info = env.step(env.action_space.sample())
    print("info keys after step:", sorted(info.keys()))
    env.close()

    for obs_type in ("pixels_agent_pos", "state"):
        e = make_env(obs_type)
        e.reset(seed=0)
        rng = np.random.RandomState(0)
        t0 = time.perf_counter()
        for _ in range(300):
            e.unwrapped.step(rng.uniform(0, 512, size=2))
        dt = time.perf_counter() - t0
        print(f"speed [{obs_type:16s}]: {300 / dt:7.1f} control steps/s  ({dt:.2f} s for 300 steps)")
        e.close()


# --------------------------------------------------------------------------- B
def part_b(cfg: dict, n_episodes: int = 10, max_steps: int = 300) -> None:
    print("\n=== B. Random agent baseline ===")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    dt = cfg["data"]["dt"]
    env = make_env()
    rows = []
    for seed in range(n_episodes):
        env.reset(seed=seed)
        env.action_space.seed(seed)
        max_cov, steps_to_success, frames = 0.0, None, []
        for step in range(1, max_steps + 1):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            max_cov = max(max_cov, float(info["coverage"]))
            if seed == 0:
                frames.append(env.render())
            if info["is_success"] and steps_to_success is None:
                steps_to_success = step
            if terminated or truncated:
                break
        rows.append({
            "seed": seed,
            "steps_run": step,
            "success": steps_to_success is not None,
            "max_coverage": max_cov,
            "steps_to_success": steps_to_success,
            "time_to_success_s": None if steps_to_success is None else steps_to_success * dt,
        })
        print(f"seed {seed}: success={rows[-1]['success']}  max_coverage={max_cov:.3f}  steps_run={step}")
        if seed == 0:
            with imageio.get_writer(VID_DIR / "random_agent_ep0.mp4", fps=10, macro_block_size=1) as w:
                for f in frames:
                    w.append_data(f)
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "random_agent.csv", index=False)
    print(f"\nsuccess rate        : {df['success'].mean() * 100:.1f}% ({int(df['success'].sum())}/{len(df)})")
    print(f"mean max_coverage   : {df['max_coverage'].mean():.3f}   (max over episodes: {df['max_coverage'].max():.3f})")
    print(f"saved: {FIG_DIR / 'random_agent.csv'} and {VID_DIR / 'random_agent_ep0.mp4'}")


# --------------------------------------------------------------------------- C
def replay_episode(env, state5, action, s, e, order, seed: int = 0) -> dict:
    """Set the dataset state at frame s, execute action[s:e], compare the T block with zarr.

    drift index i = discrepancy after i executed actions (i = 0 is right after set_env_state).
    """
    env.reset(seed=seed)
    set_env_state(env, state5[s], order)
    pose0 = get_block_pose(env)
    res = rollout_targets(env, action[s:e].astype(np.float64))
    sim = np.vstack([pose0[None], res["block_poses"][:-1]])  # [L, 3]
    ref = state5[s:e, 2:5].astype(np.float64)
    res["drift_pos"] = np.linalg.norm(sim[:, :2] - ref[:, :2], axis=1)
    res["drift_ang"] = np.degrees(np.abs(wrap_angle(sim[:, 2] - ref[:, 2])))
    res["demo_len"] = e - s
    return res


def part_c(cfg: dict) -> None:
    print("\n=== C. Replay demo on val episodes ===")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cfg["data"]["cache_dir"])
    state5 = np.asarray(cache["state5"])
    action = np.asarray(cache["action"])
    bounds = episode_bounds(cache["episode_ends"])
    val = cache["split"]["val"]
    dt = cfg["data"]["dt"]
    env = make_env("state")

    # 1) choose the set-state order on the first val episode
    ep0 = val[0]
    s0, e0 = bounds[ep0]
    k = min(20, int(e0 - s0) - 1)
    best_order, best_score = None, 1e18
    print(f"order selection on val episode {ep0} (drift at step 0 and step {k}):")
    for order in ("angle_first", "position_first"):
        r = replay_episode(env, state5, action, s0, e0, order)
        score = r["drift_pos"][0] + r["drift_pos"][k]
        print(f"  {order:15s} drift_pos[0]={r['drift_pos'][0]:8.3f} px | drift_pos[{k}]={r['drift_pos'][k]:8.3f} px"
              f" | success={r['success_step'] is not None}")
        if score < best_score:
            best_order, best_score = order, score
    print(f"=> chosen order: {best_order}   (config currently: {cfg['sim']['set_state_order']})")
    if best_order != cfg["sim"]["set_state_order"]:
        print("!! Update sim.set_state_order in configs/default.yaml to the chosen order.")

    # 2) replay all val episodes
    rows, P, A = [], [], []
    for ep in val:
        s, e = bounds[ep]
        r = replay_episode(env, state5, action, s, e, best_order)
        ss = r["success_step"]
        rows.append({
            "episode": ep, "demo_len": r["demo_len"], "success": ss is not None,
            "max_coverage": float(r["coverage"].max()), "steps_to_success": ss,
            "time_to_success_s": None if ss is None else ss * dt,
            "drift_pos_0": r["drift_pos"][0], "drift_pos_20": r["drift_pos"][min(20, r["demo_len"] - 1)],
            "drift_pos_final": r["drift_pos"][-1], "drift_ang_final": r["drift_ang"][-1],
        })
        P.append(r["drift_pos"])
        A.append(r["drift_ang"])
        d = rows[-1]
        print(f"ep {ep:3d}: len={d['demo_len']:3d} success={str(d['success']):5s} max_cov={d['max_coverage']:.3f}"
              f" steps={d['steps_to_success']} drift0={d['drift_pos_0']:.2f} drift20={d['drift_pos_20']:.1f}"
              f" final={d['drift_pos_final']:.1f}px")
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "replay_val.csv", index=False)
    t = df["time_to_success_s"].dropna()
    print("\n--- summary ---")
    print(f"success rate: {df['success'].mean() * 100:.1f}% ({int(df['success'].sum())}/{len(df)})")
    print(f"mean max_coverage: {df['max_coverage'].mean():.3f}")
    if len(t):
        print(f"time to success (s): mean={t.mean():.2f} median={t.median():.2f} "
              f"p25={np.percentile(t, 25):.2f} p75={np.percentile(t, 75):.2f}")
    print(f"drift_pos at step 0 : mean={df['drift_pos_0'].mean():.3f} px")
    print(f"drift_pos at step 20: mean={df['drift_pos_20'].mean():.2f} px  (target < 20)")

    # 3) figure: mean +- std (black/white friendly)
    max_len = max(len(p) for p in P)
    PP = np.full((len(P), max_len), np.nan)
    AA = np.full((len(P), max_len), np.nan)
    for i, (p, a) in enumerate(zip(P, A)):
        PP[i, : len(p)] = p
        AA[i, : len(a)] = a
    ok = (~np.isnan(PP)).sum(0) >= 5
    x = np.arange(max_len)[ok]
    pm, ps = np.nanmean(PP[:, ok], 0), np.nanstd(PP[:, ok], 0)
    am = np.nanmean(AA[:, ok], 0)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, pm, "-", c="black", lw=1.8, label="position drift (px, left)")
    ax1.fill_between(x, pm - ps, pm + ps, color="gray", alpha=0.25)
    ax1.set_xlabel("control step (0.1 s)")
    ax1.set_ylabel("T-block position drift (px)")
    ax2 = ax1.twinx()
    ax2.plot(x, am, "--", c="black", lw=1.8, label="angle drift (deg, right)")
    ax2.set_ylabel("T-block angle drift (deg)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-8, 1.0)
    ax2.set_ylim(0, 1.0)
    ax1.set_title(f"Replay drift vs dataset ({len(P)} val episodes): max {np.nanmax(PP):.0e} px, {np.nanmax(AA):.0e} deg")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "replay_drift.png", dpi=150)
    plt.close(fig)
    print(f"saved: {FIG_DIR / 'replay_val.csv'} and {FIG_DIR / 'replay_drift.png'}")


# --------------------------------------------------------------------------- D
def part_d(cfg: dict, n_frames: int = 50) -> None:
    print("\n=== D. Sim render vs dataset image ===")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cfg["data"]["cache_dir"])
    images, state5 = cache["images"], np.asarray(cache["state5"])
    bounds = episode_bounds(cache["episode_ends"])
    order = cfg["sim"]["set_state_order"]
    env = make_env("pixels_agent_pos")

    def render_at(i: int) -> np.ndarray:
        env.reset(seed=0)
        return set_env_state(env, state5[i], order)["pixels"]

    sim0 = render_at(0).astype(np.float32)
    ds0 = np.asarray(images[0]).astype(np.float32)
    d0 = np.abs(sim0 - ds0)
    print(f"frame 0 (episode 0): mean|diff| = {d0.mean():.3f}   max|diff| = {d0.max():.0f}   (0..255)")
    print(f"RGB order check: as-is = {d0.mean():.3f} | R<->B swapped = {np.abs(sim0[..., ::-1] - ds0).mean():.3f}")

    rng = np.random.RandomState(0)
    val_idx = np.concatenate([np.arange(*bounds[ep]) for ep in cache["split"]["val"]])
    pick = rng.choice(val_idx, size=n_frames, replace=False)
    diffs = np.array([np.abs(render_at(int(i)).astype(np.float32) - np.asarray(images[int(i)]).astype(np.float32)).mean()
                      for i in pick])
    print(f"{n_frames} random val frames: mean|diff| mean={diffs.mean():.3f}  median={np.median(diffs):.3f}  max={diffs.max():.3f}")
    env.close()

    dm = d0.mean(-1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(sim0.astype(np.uint8))
    axes[0].set_title("sim render (set_env_state)")
    axes[1].imshow(ds0.astype(np.uint8))
    axes[1].set_title("dataset images.npy[0]")
    im = axes[2].imshow(dm, cmap="gray_r", vmin=0, vmax=max(1.0, float(dm.max())))
    axes[2].set_title(f"|diff| (mean {d0.mean():.2f} / 255)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "render_vs_dataset.png", dpi=150)
    plt.close(fig)
    print("saved:", FIG_DIR / "render_vs_dataset.png")


# --------------------------------------------------------------------------- E
def part_e(cfg: dict) -> None:
    print("\n=== E. Replay video (best val episode) ===")
    VID_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIG_DIR / "replay_val.csv"
    if not csv_path.exists():
        raise SystemExit("Run part C first (needs replay_val.csv)")
    df = pd.read_csv(csv_path)
    ep = int(df.loc[df["max_coverage"].idxmax(), "episode"])

    cache = load_cache(cfg["data"]["cache_dir"])
    state5, action = np.asarray(cache["state5"]), np.asarray(cache["action"])
    s, e = episode_bounds(cache["episode_ends"])[ep]
    dt = cfg["data"]["dt"]

    env = make_env("pixels_agent_pos")
    env.reset(seed=0)
    set_env_state(env, state5[s], cfg["sim"]["set_state_order"])
    frames = [env.render()]
    covs = [get_coverage(env)]
    res = rollout_targets(env, action[s:e].astype(np.float64), record=True)
    frames += res["frames"]
    covs += list(res["coverage"])
    env.close()

    annotated = []
    for k, (f, c) in enumerate(zip(frames, covs)):
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 360, 28], fill=(255, 255, 255))
        draw.text((8, 7), f"expert replay | ep {ep} | t = {k * dt:4.1f} s | coverage = {c:.3f}", fill=(0, 0, 0))
        annotated.append(img)

    mp4 = VID_DIR / f"replay_ep{ep}.mp4"
    with imageio.get_writer(mp4, fps=10, macro_block_size=1) as w:
        for img in annotated:
            w.append_data(np.asarray(img))
    gif = Path("media") / "expert_replay.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    small = [img.resize((340, 340), Image.BILINEAR) for img in annotated]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=100, loop=0)
    print(f"episode {ep}: {len(frames)} frames ({len(frames) * dt:.1f} s), final coverage = {covs[-1]:.3f}, "
          f"max = {max(covs):.3f}")
    print(f"saved: {mp4} and {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--parts", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg["project"]["seed"])
    for p in args.parts:
        PARTS[p.upper()](cfg)


if __name__ == "__main__":
    main()

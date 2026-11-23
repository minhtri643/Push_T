"""Pipeline 5 (evaluation): videos of the system for the README (GIF) and the release (MP4).

    python scripts/5_evaluation/make_video.py grid       # final system on 6 benchmark starts -> docs/media/final_policy.gif
    python scripts/5_evaluation/make_video.py compare    # E0 baseline vs final, same start   -> docs/media/baseline_vs_final.gif

Every tile replays a benchmark episode exactly: same start, same random seeds, same calls as
scripts/5_evaluation/benchmark.py. The script checks that the maximum coverage of each replay equals the number
stored in results/benchmark/<label>.json, so the video shows the measured episodes, not hand-picked new ones
(the choice of WHICH starts to show is printed and explained below).

Drawn on each frame (world coordinates, 512 x 512 workspace):
    green T       the goal                    grey T     the block (simulator state)
    blue dot      the pusher                  blue line  the block path planned by the DiT
    orange T      where the World Model imagines the block after the actions about to run
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pusht_wm_dit.envs.pusht_sim import PushTSim  # noqa: E402
from pusht_wm_dit.paths import load_config  # noqa: E402
from pusht_wm_dit.policy import SUCCESS_THRESHOLD, build_policy, postprocess_action, resolve_device  # noqa: E402

TEE = np.array([(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)], np.float64)
COL = {"bg": (247, 247, 245), "wall": (60, 60, 60), "goal": (144, 214, 144), "goal_edge": (60, 150, 60),
       "block": (120, 132, 150), "block_edge": (55, 62, 75), "agent": (66, 105, 225), "path": (52, 87, 213),
       "ghost": (240, 140, 30), "text": (30, 30, 30), "ok": (35, 128, 74)}


def tee(x, y, th, scale):
    c, s = math.cos(th), math.sin(th)
    pts = np.stack([x + c * TEE[:, 0] - s * TEE[:, 1], y + s * TEE[:, 0] + c * TEE[:, 1]], -1)
    return np.round(pts * scale * 8).astype(np.int32)        # 3 fractional bits for cv2 shift=3


def draw(obs, size, path=None, ghost=None, title="", step=0, cov=0.0, peak=0.0, success_at=None):
    """RGB frame (size x size + caption) of the current state, drawn at 2x and downsampled (anti-aliased)."""
    S = 2 * size
    k = S / 512.0
    img = np.full((S, S, 3), COL["bg"], np.uint8)
    cv2.fillPoly(img, [tee(256, 256, math.pi / 4, k)], COL["goal"], cv2.LINE_AA, shift=3)
    cv2.polylines(img, [tee(256, 256, math.pi / 4, k)], True, COL["goal_edge"], 2, cv2.LINE_AA, shift=3)
    bx, by, bth = obs["block_pose"]
    cv2.fillPoly(img, [tee(bx, by, bth, k)], COL["block"], cv2.LINE_AA, shift=3)
    cv2.polylines(img, [tee(bx, by, bth, k)], True, COL["block_edge"], 2, cv2.LINE_AA, shift=3)
    if path is not None:
        pts = np.round(np.stack([path[:, 0] * 256 + 256, path[:, 1] * 256 + 256], -1) * k * 8).astype(np.int32)
        cv2.polylines(img, [pts], False, COL["path"], 3, cv2.LINE_AA, shift=3)
    if ghost is not None:
        cv2.polylines(img, [tee(*ghost, k)], True, COL["ghost"], 4, cv2.LINE_AA, shift=3)
    ax, ay = obs["agent_pos_px"]
    cv2.circle(img, (int(ax * k * 8), int(ay * k * 8)), int(15 * k * 8), COL["agent"], -1, cv2.LINE_AA, shift=3)
    cv2.rectangle(img, (2, 2), (S - 3, S - 3), COL["wall"], 4)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    f = size / 240.0
    cap = np.full((int(58 * f), size, 3), 255, np.uint8)
    put = lambda s, y, color, scale=0.42: cv2.putText(cap, s, (int(6 * f), int(y * f)), cv2.FONT_HERSHEY_SIMPLEX,  # noqa: E731
                                                      scale * f, color, 1, cv2.LINE_AA)
    put(title, 16, COL["text"])
    put(f"coverage {cov * 100:5.1f}%   max {peak * 100:5.1f}%", 34, COL["text"])
    put(f"SUCCESS at step {success_at}" if success_at else f"step {step}", 52, COL["ok"] if success_at else COL["text"])
    return np.concatenate([img, cap], 0)


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark", os.path.join("scripts", "5_evaluation", "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay(cfg, kind, ckpt_dir, k, start, size, title, max_steps, exec_h, offset=0, vision_backend=None):
    """Replays benchmark episode k exactly (same seeds and calls as benchmark.run + run_episode); returns frames, max cov."""
    policy = build_policy(cfg, resolve_device(cfg), ckpt_dir=ckpt_dir, kind=kind, vision_backend=vision_backend)
    sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=tuple(cfg["env"]["render_size"]), max_steps=max_steps)
    torch.manual_seed(10_000 + k + 1000 * offset)
    np.random.seed(10_000 + k + 1000 * offset)
    _, _, init = start
    if "init_state" in init:
        sim.reset(seed=init["seed"])
        obs = sim.set_state(init["init_state"][:2], init["init_state"][2:5])
    else:
        obs, _ = sim.reset(seed=init["seed"])
    if hasattr(policy, "reset"):
        policy.reset(obs)
    peak, step, success_at = obs["coverage"], 0, None
    frames = [draw(obs, size, title=title, cov=obs["coverage"], peak=peak)]
    done = obs["coverage"] > SUCCESS_THRESHOLD
    while not done and step < max_steps:
        actions, path = policy.plan(obs)
        chunk = actions[:exec_h]
        with torch.no_grad():
            imag = policy.imagine(obs, chunk) if hasattr(policy, "imagine") else None
        ghost = tuple(imag[-1]) if imag is not None and len(imag) else None
        for a_idx in range(min(exec_h, len(actions))):
            act = postprocess_action(actions[a_idx], obs)
            obs, _, terminated, truncated, _ = sim.step(act, normalized_action=True)
            if hasattr(policy, "observe"):
                policy.observe(obs)
            step += 1
            peak = max(peak, obs["coverage"])
            if terminated or obs["coverage"] > SUCCESS_THRESHOLD:
                done, success_at = True, step
            elif truncated or step >= max_steps:
                done = True
            frames.append(draw(obs, size, path=path, ghost=ghost, title=title, step=step, cov=obs["coverage"], peak=peak, success_at=success_at))
            if done:
                break
    sim.close()
    return frames, peak


def tile(rows):
    """rows: list of lists of frame sequences -> one sequence of mosaics (shorter sequences hold their last frame)."""
    n = max(len(s) for r in rows for s in r)
    out = []
    for t in range(n):
        out.append(np.concatenate([np.concatenate([s[min(t, len(s) - 1)] for s in r], 1) for r in rows], 0))
    return out


def write(frames, gif, mp4, every, fps, hold=20):
    frames = frames[::every] + [frames[-1]] * hold
    os.makedirs(os.path.dirname(gif), exist_ok=True)
    os.makedirs(os.path.dirname(mp4), exist_ok=True)
    imageio.mimsave(mp4, frames, fps=fps, quality=8, macro_block_size=1)
    imageio.mimsave(gif, frames, duration=1000.0 / fps, loop=0)
    print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB, {len(frames)} frames) and {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB)")


def check(label, k, peak):
    """On a GPU the replay must reproduce the stored benchmark episode exactly. On a CPU the random numbers of the
    diffusion sampler differ from the GPU's (same seed, different generator), so the replay is another noise draw:
    the video is still written, with a warning."""
    ref = json.load(open(os.path.join("results", "benchmark", f"{label}.json"), encoding="utf-8"))["episodes"][k]["max_coverage"]
    ok = abs(ref - peak) < 1e-9
    print(f"  start {k:2d} ({label}): replay max coverage {peak:.4f} vs benchmark {ref:.4f} -> {'IDENTICAL' if ok else 'DIFFERENT'}")
    if not ok and torch.cuda.is_available():
        sys.exit("replay does not reproduce the benchmark episode")
    if not ok:
        print("  (CPU run: diffusion noise differs from the GPU benchmark, so this replay is a different noise draw)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("grid", "compare"))
    p.add_argument("--size", type=int, default=240, help="tile size in pixels (GIF); the MP4 uses the same frames")
    p.add_argument("--every", type=int, default=2, help="keep every n-th simulator step (10 steps = 1 s)")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    cfg = load_config()
    bench = load_benchmark()
    starts = bench.starts()
    max_steps, exec_h = cfg["env"]["max_steps"], cfg["evaluation"]["exec_horizon"]
    if args.mode == "grid":
        label = "E4_final__dit_wm"
        eps = json.load(open(f"results/benchmark/{label}.json", encoding="utf-8"))["episodes"]
        # Which starts: the first two SUCCESSFUL episodes of each suite, in benchmark order (no cherry-picking by
        # appearance). The success rate itself (76%) is printed in the caption of the README, not implied by the video.
        pick = []
        for suite in ("seeds50", "hard20", "random25"):
            pick += [k for k, e in enumerate(eps) if e["suite"] == suite and e["success"]][:2]
        print("starts shown:", [(k, starts[k][0], starts[k][1]) for k in pick])
        seqs = []
        for k in pick:
            fr, peak = replay(cfg, "dit_wm", None, k, starts[k], args.size, f"{starts[k][0]} | {starts[k][1]}", max_steps, exec_h)
            check(label, k, peak)
            seqs.append(fr)
        write(tile([seqs[0:3], seqs[3:6]]), "docs/media/final_policy.gif", "results/videos/final_policy.mp4", args.every, args.fps)
    else:
        a_label, b_label = "E0_baseline__dit_id", "E4_final__dit_wm"
        ea = json.load(open(f"results/benchmark/{a_label}.json", encoding="utf-8"))["episodes"]
        eb = json.load(open(f"results/benchmark/{b_label}.json", encoding="utf-8"))["episodes"]
        # The first start (benchmark order) where the baseline fails and the final system succeeds.
        k = next(i for i, (x, y) in enumerate(zip(ea, eb)) if not x["success"] and y["success"])
        print("start shown:", k, starts[k][0], starts[k][1])
        fa, pa = replay(cfg, "dit_id", "checkpoints/E0_baseline", k, starts[k], args.size, "E0 original baseline", max_steps, exec_h)
        check(a_label, k, pa)
        fb, pb = replay(cfg, "dit_wm", None, k, starts[k], args.size, "E4 final system", max_steps, exec_h)
        check(b_label, k, pb)
        write(tile([[fa, fb]]), "docs/media/baseline_vs_final.gif", "results/videos/baseline_vs_final.mp4", args.every, args.fps)


if __name__ == "__main__":
    main()

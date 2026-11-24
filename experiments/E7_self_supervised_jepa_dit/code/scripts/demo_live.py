"""Live closed-loop demo: frozen JEPA world model + DiT policy playing PushT on endless random seeds.

Window + endless  : python -m scripts.demo_live --continuous
5 episodes        : python -m scripts.demo_live --n_episodes 5
No window (server): python -m scripts.demo_live --no_window --n_episodes 20 --save_dir media/live
Quit the window with q / Esc, or Ctrl+C in the terminal. Every seed is logged, failures included.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from src.eval_sim import PushTAgent, draw_overlay, load_models
from src.sim_env import make_env
from src.utils import get_device, load_config, set_seed


def add_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2
    img = np.ascontiguousarray(frame.copy())
    h = 8 + 18 * len(lines)
    cv2.rectangle(img, (0, 24), (430, 24 + h), (255, 255, 255), -1)
    for k, text in enumerate(lines):
        cv2.putText(img, text, (6, 24 + 18 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm_run", default="wm_main")
    ap.add_argument("--policy_run", default="dit_main")
    ap.add_argument("--start_seed", type=int, default=1000, help="fresh seeds, disjoint from the 100000+ eval set")
    ap.add_argument("--continuous", action="store_true", help="never stop (Ctrl+C or q to quit)")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_infer", type=int, default=8, help="DDIM steps (8 = same quality as 16, half latency)")
    ap.add_argument("--n_action_steps", type=int, default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1.0 = real time (10 Hz)")
    ap.add_argument("--hold_ms", type=int, default=1200, help="freeze the last frame of each episode")
    ap.add_argument("--no_window", action="store_true")
    ap.add_argument("--save_dir", default=None, help="save one mp4 per seed here")
    ap.add_argument("--log", default="outputs/demo/live_log.csv")
    args = ap.parse_args()

    show = not args.no_window
    if show:
        try:
            import cv2
            cv2.namedWindow("PushT live demo", cv2.WINDOW_AUTOSIZE)
        except Exception as e:  # headless OpenCV build
            raise SystemExit(f"Cannot open a window ({e}). Re-run with --no_window --save_dir media/live")
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    render = show or save_dir is not None

    device = get_device()
    wm, policy, cfg = load_models(args.wm_run, args.policy_run, device)
    set_seed(cfg["project"]["seed"])
    max_steps = int(load_config(args.config)["eval"]["max_steps"])
    agent = PushTAgent(wm, policy, cfg, device, args.n_action_steps, args.n_infer)
    dt = agent.dt
    env = make_env("pixels_agent_pos")

    obs, _ = env.reset(seed=0)                      # warm-up: first CUDA call is slow
    agent.act(obs)
    agent.reset()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["seed", "success", "steps", "max_coverage", "infer_p95_ms", "end"])

    i, n_ok, covs, quit_ = 0, 0, [], False
    print(f"live demo | DDIM {args.n_infer} steps | n_action_steps {agent.n_action_steps} | seeds from {args.start_seed}")
    try:
        while (args.continuous or i < args.n_episodes) and not quit_:
            seed = args.start_seed + i
            obs, _ = env.reset(seed=seed)
            agent.reset(seed)
            max_cov, s095, frames, step = 0.0, None, [], 0
            for step in range(1, max_steps + 1):
                t_loop = time.perf_counter()
                obs, _, term, trunc, info = env.step(agent.act(obs).astype(np.float32))
                cov = float(info["coverage"])
                max_cov = max(max_cov, cov)
                if s095 is None and info["is_success"]:
                    s095 = step
                if render:
                    fr = draw_overlay(env.render(), agent.last_path, agent.n_action_steps, step, cov, dt)
                    fr = add_hud(fr, [f"seed {seed} | episode {i + 1} | max coverage {max_cov:.3f}",
                                      f"so far: {n_ok}/{i} success @0.95  (target 0.95)"])
                    if save_dir:
                        frames.append(fr)
                    if show:
                        cv2.imshow("PushT live demo", fr[:, :, ::-1])
                        wait = dt / args.speed - (time.perf_counter() - t_loop)
                        key = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
                        if key in (27, ord("q")):
                            quit_ = True
                            break
                if term or trunc:
                    break
            if quit_:
                break
            if show:
                cv2.waitKey(args.hold_ms)
            ok = s095 is not None
            n_ok += int(ok)
            covs.append(max_cov)
            i += 1
            p95 = float(np.percentile(agent.infer_ms, 95)) if agent.infer_ms else float("nan")
            end = "success" if ok else "timeout"
            print(f"Seed {seed}: success={ok} steps={step} max_coverage={max_cov:.3f} infer_p95={p95:.0f}ms end={end}"
                  f"   | running {n_ok}/{i} = {100 * n_ok / i:.0f}%  mean max cov {np.mean(covs):.3f}")
            writer.writerow([seed, ok, step, f"{max_cov:.4f}", f"{p95:.1f}", end])
            log_f.flush()
            if save_dir and frames:
                with imageio.get_writer(save_dir / f"seed{seed}_{end}.mp4", fps=10, macro_block_size=1) as w:
                    for f in frames:
                        w.append_data(f)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        env.close()
        log_f.close()
        if show:
            cv2.destroyAllWindows()
    if i:
        print(f"\n=== LIVE SUMMARY: {n_ok}/{i} success@0.95 ({100 * n_ok / i:.0f}%), mean max coverage {np.mean(covs):.3f} ===")
        print(f"log appended to {log_path}")


if __name__ == "__main__":
    main()

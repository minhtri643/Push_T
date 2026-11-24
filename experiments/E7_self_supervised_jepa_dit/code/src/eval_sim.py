"""Closed-loop evaluation of the frozen JEPA world model + DiT policy in gym-pusht.

Domain-gap check : python -m src.eval_sim --wm_run wm_main --policy_run dit_main --domain_check
Smoke test       : python -m src.eval_sim --wm_run wm_main --policy_run dit_main --n_episodes 5 --n_videos 2 --tag smoke
Full evaluation  : python -m src.eval_sim --wm_run wm_main --policy_run dit_main
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src.dataset import episode_bounds, load_cache
from src.dit_policy import DiffusionPolicy, path_to_velocity, velocity_to_targets
from src.sim_env import make_env, set_env_state
from src.utils import get_device, load_config, set_seed
from src.world_model import JEPAWorldModel


# --------------------------------------------------------------------------- agent
class PushTAgent:
    """Frozen JEPA world model + EMA DiT policy: pixels -> target pusher position."""

    def __init__(self, wm, policy, cfg, device, n_action_steps=None, n_infer=None, seed=0):
        self.wm, self.policy, self.device = wm, policy, device
        self.dt = float(cfg["data"]["dt"])
        self.ws = float(cfg["data"]["workspace_size"])
        self.n_action_steps = int(n_action_steps or cfg["policy"]["n_action_steps"])
        self.n_infer = int(n_infer or cfg["policy"]["n_inference_steps"])
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.frames = deque(maxlen=2)
        self.queue = deque()
        self.last_path = None
        self.infer_ms: list[float] = []
        if seed is not None:
            self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        pix = obs["pixels"]
        if not self.frames:
            self.frames.append(pix)
        self.frames.append(pix)                                  # first call: the frame is repeated twice
        if not self.queue:
            t0 = time.perf_counter()
            fr = torch.from_numpy(np.stack(self.frames)[None]).to(self.device)       # [1, 2, 96, 96, 3]
            s_t = self.wm.encode_state(fr)                                           # [1, 256]
            pos = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))[None].to(self.device)
            path = self.policy.sample(s_t, pos, self.n_infer, generator=self.gen)   # [1, H, 2] px
            v = path_to_velocity(path, pos, self.dt)
            targets = velocity_to_targets(v, pos, self.dt)[0, : self.n_action_steps].clamp(0, self.ws)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.infer_ms.append((time.perf_counter() - t0) * 1000)
            self.last_path = path[0].cpu().numpy()
            self.queue.extend(targets.cpu().numpy())
        return self.queue.popleft().astype(np.float64)


def load_models(wm_run, policy_run, device, out_root="outputs/runs", wm_random_init=False):
    pk = torch.load(Path(out_root) / policy_run / "ckpt" / "best.pt", map_location=device, weights_only=False)
    cfg = pk["cfg"]
    if wm_random_init:
        set_seed(123)                                            # same init as export_latents (random-init latents)
        wm = JEPAWorldModel(cfg["wm"])
    else:
        wk = torch.load(Path(out_root) / wm_run / "ckpt" / "best.pt", map_location=device, weights_only=False)
        wm = JEPAWorldModel(wk["cfg"]["wm"])
        wm.load_state_dict(wk["model"])
    wm.to(device).eval().requires_grad_(False)
    policy = DiffusionPolicy(cfg["policy"], wm.s_dim, cfg["data"]["workspace_size"]).to(device)
    policy.load_state_dict(pk["model"])
    policy.eval()
    return wm, policy, cfg


# --------------------------------------------------------------------------- domain gap
@torch.no_grad()
def domain_check(wm, cfg, device, latents_path, n: int = 50) -> float:
    """S_t from sim-rendered frames vs S_t stored in the exported latents (clean version)."""
    latents = np.load(latents_path)
    cache = load_cache(cfg["data"]["cache_dir"])
    state5 = np.asarray(cache["state5"], dtype=np.float64)
    bounds = episode_bounds(cache["episode_ends"])
    ts = np.concatenate([np.arange(bounds[e][0] + 1, bounds[e][1]) for e in cache["split"]["val"]])
    picks = np.sort(np.random.RandomState(0).choice(ts, size=n, replace=False))
    env = make_env("pixels_agent_pos")
    order = cfg["sim"]["set_state_order"]
    cos = []
    for t in picks:
        env.reset(seed=0)
        prev = set_env_state(env, state5[t - 1], order)["pixels"]
        cur = set_env_state(env, state5[t], order)["pixels"]
        fr = torch.from_numpy(np.stack([prev, cur])[None]).to(device)
        s_sim = wm.encode_state(fr)[0].cpu().numpy()
        ref = latents[0, t]
        cos.append(float(s_sim @ ref / (np.linalg.norm(s_sim) * np.linalg.norm(ref) + 1e-8)))
    env.close()
    cos = np.array(cos)
    print(f"cosine(S_sim, S_dataset) over {n} val frames: mean={cos.mean():.4f}  p10={np.percentile(cos, 10):.4f}"
          f"  min={cos.min():.4f}")
    print(f"[{'PASS' if cos.mean() > 0.9 else 'FAIL'}] mean cosine > 0.9")
    return float(cos.mean())


# --------------------------------------------------------------------------- evaluation
def draw_overlay(frame: np.ndarray, path: np.ndarray, n_exec: int, step: int, cov: float, dt: float) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    s = frame.shape[0] / 512.0
    pts = [(float(x * s), float(y * s)) for x, y in path]
    d.line(pts, fill=(255, 0, 0), width=2)
    d.line(pts[:n_exec], fill=(255, 0, 255), width=4)
    d.rectangle([0, 0, 310, 22], fill=(255, 255, 255))
    d.text((6, 5), f"t = {step * dt:4.1f} s   coverage = {cov:.3f}", fill=(0, 0, 0))
    return np.asarray(img)


def _tstats(x: pd.Series) -> dict | None:
    x = x.dropna()
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))}


def evaluate(agent: PushTAgent, cfg: dict, out_dir: Path, n_episodes=50, seed_start=100000,
             n_videos=3, max_steps=300) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    env = make_env("pixels_agent_pos")
    dt = agent.dt
    rows = []
    for i in range(n_episodes):
        seed = seed_start + i
        obs, _ = env.reset(seed=seed)
        agent.reset(seed)
        frames, max_cov, cov, s095, s090 = [], 0.0, 0.0, None, None
        for step in range(1, max_steps + 1):
            obs, _, term, trunc, info = env.step(agent.act(obs).astype(np.float32))
            cov = float(info["coverage"])
            max_cov = max(max_cov, cov)
            if s090 is None and cov > 0.90:
                s090 = step
            if s095 is None and info["is_success"]:
                s095 = step
            if i < n_videos:
                frames.append(draw_overlay(env.render(), agent.last_path, agent.n_action_steps, step, cov, dt))
            if term or trunc:
                break
        rows.append({
            "seed": seed, "success": s095 is not None, "steps_to_success": s095,
            "time_to_success_s": None if s095 is None else s095 * dt,
            "success_090": s090 is not None, "steps_to_090": s090,
            "time_to_090_s": None if s090 is None else s090 * dt,
            "max_coverage": max_cov, "final_coverage": cov, "steps_run": step,
            "mean_infer_ms": float(np.mean(agent.infer_ms)),
        })
        print(f"[{i + 1:3d}/{n_episodes}] seed {seed}: success={s095 is not None}  max_cov={max_cov:.3f}  "
              f"steps={step}  infer={rows[-1]['mean_infer_ms']:.0f} ms")
        if i < n_videos:
            with imageio.get_writer(out_dir / "videos" / f"ep{seed}.mp4", fps=10, macro_block_size=1) as w:
                for f in frames:
                    w.append_data(f)
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "eval_episodes.csv", index=False)
    summary = {
        "n_episodes": n_episodes, "success_rate": float(df["success"].mean()),
        "success_090_rate": float(df["success_090"].mean()),
        "mean_max_coverage": float(df["max_coverage"].mean()),
        "median_max_coverage": float(df["max_coverage"].median()),
        "mean_final_coverage": float(df["final_coverage"].mean()),
        "time_to_success_s": _tstats(df["time_to_success_s"]), "time_to_090_s": _tstats(df["time_to_090_s"]),
        "mean_infer_ms": float(df["mean_infer_ms"].mean()),
        "n_action_steps": agent.n_action_steps, "n_infer": agent.n_infer,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(f"success@0.95        : {summary['success_rate'] * 100:.1f}% ({int(df['success'].sum())}/{n_episodes})")
    print(f"success@0.90        : {summary['success_090_rate'] * 100:.1f}%   (expert replay on val: 10/20 = 50%)")
    print(f"mean max coverage   : {summary['mean_max_coverage']:.3f}   (expert replay: 0.898 | LeRobot DP: 0.955)")
    print(f"time to success (s) : {summary['time_to_success_s']}")
    print(f"mean inference time : {summary['mean_infer_ms']:.0f} ms per path")
    print(f"saved in {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm_run", default="wm_main")
    ap.add_argument("--policy_run", default="dit_main")
    ap.add_argument("--latents", default="data/cache/latents_wm_main.npy")
    ap.add_argument("--wm_random_init", action="store_true")
    ap.add_argument("--domain_check", action="store_true")
    ap.add_argument("--n_episodes", type=int, default=None)
    ap.add_argument("--n_videos", type=int, default=None)
    ap.add_argument("--n_action_steps", type=int, default=None)
    ap.add_argument("--n_infer", type=int, default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    device = get_device()
    wm, policy, cfg = load_models(args.wm_run, args.policy_run, device, wm_random_init=args.wm_random_init)
    set_seed(cfg["project"]["seed"])
    if args.domain_check:
        domain_check(wm, cfg, device, args.latents)
        return

    ev = load_config(args.config)["eval"]
    agent = PushTAgent(wm, policy, cfg, device, args.n_action_steps, args.n_infer)
    out_dir = Path("outputs/runs") / args.policy_run / "eval" / args.tag
    evaluate(agent, cfg, out_dir,
             n_episodes=args.n_episodes or ev["n_episodes"], seed_start=ev["seed_start"],
             n_videos=args.n_videos if args.n_videos is not None else ev["n_videos"],
             max_steps=ev["max_steps"])


if __name__ == "__main__":
    main()

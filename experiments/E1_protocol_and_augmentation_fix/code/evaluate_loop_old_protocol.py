"""Stages E0-E1 (old 50-episode protocol, kept for reference): Closed-Loop Receding Horizon Evaluation & Hard-Case Logging.

Closed-loop pipeline (see pusht_wm_dit/policy.py):
1. Observe RGB frame (96x96x3) and agent position.
2. Encode into latent s_t with the frozen World Model.
3. Plan the H=16 block trajectory with the DiT (step-0 inpainting, best-of-8).
4. Produce Ta actions with the Inverse Dynamics controller and execute them (RHC).

Standard Push-T protocol: episodes run until success (coverage > 0.95) or max_steps,
and the reported metrics are max coverage per episode and success rate.
Heuristics outside the learned policy are opt-in: --stance, --early-stop.

Examples:
    python experiments/E1_protocol_and_augmentation_fix/code/evaluate_loop_old_protocol.py
    python experiments/E1_protocol_and_augmentation_fix/code/evaluate_loop_old_protocol.py --ckpt-dir checkpoints/E0_baseline --out-dir results/legacy_protocol/E0_baseline/standard
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json

import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim, VideoRecorder
from pusht_wm_dit.policy import build_policy, resolve_device, run_episode, summarize


def parse_args(config):
    eval_cfg, env_cfg = config["evaluation"], config["env"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=eval_cfg["num_eval_episodes"])
    p.add_argument("--max-steps", type=int, default=env_cfg["max_steps"])
    p.add_argument("--exec-horizon", type=int, default=eval_cfg["exec_horizon"])
    p.add_argument("--ckpt-dir", default=None, help="Override checkpoint directory (file names kept)")
    p.add_argument("--out-dir", default=eval_cfg["video_dir"])
    p.add_argument("--stance", action="store_true", help="Enable stance-navigation heuristic")
    p.add_argument("--early-stop", action="store_true", help="Enable target-lock heuristic")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def main():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    args = parse_args(config)
    device = resolve_device(config)
    env_cfg = config["env"]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print(f"STEP 5: CLOSED-LOOP EVALUATION (DEVICE: {device.upper()})")
    print(f"  checkpoints: {args.ckpt_dir or 'from config'} | episodes: {args.episodes} | max_steps: {args.max_steps}")
    print(f"  exec_horizon: {args.exec_horizon} | stance: {args.stance} | early_stop: {args.early_stop}")
    print("=" * 70)

    policy = build_policy(config, device, ckpt_dir=args.ckpt_dir)
    sim = PushTSim(
        env_name=env_cfg["env_name"],
        render_size=tuple(env_cfg["render_size"]),
        max_steps=args.max_steps,
    )
    video_path = os.path.join(args.out_dir, "eval_rollout.mp4")
    recorder = None if args.no_video else VideoRecorder(output_path=video_path, fps=15, resolution=(384, 384))

    results = []
    for ep_idx in range(args.episodes):
        ep_seed = config["seed"] + ep_idx * 101
        res = run_episode(
            sim,
            policy,
            seed=ep_seed,
            max_steps=args.max_steps,
            exec_horizon=args.exec_horizon,
            use_stance_heuristic=args.stance,
            early_stop=args.early_stop,
            recorder=recorder if ep_idx == 0 else None,
        )
        results.append(res)
        print(
            f"  Episode {ep_idx + 1:03d}/{args.episodes:03d} (seed {ep_seed}) | "
            f"max cov {res['max_coverage'] * 100:5.1f}% | final cov {res['final_coverage'] * 100:5.1f}% | "
            f"steps {res['steps']:3d} | {res['end_reason']}"
        )

    sim.close()
    if recorder is not None:
        recorder.save()

    summary = summarize(results)
    settings = {
        "ckpt_dir": args.ckpt_dir,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "exec_horizon": args.exec_horizon,
        "stance_heuristic": args.stance,
        "early_stop": args.early_stop,
        "success_definition": "max coverage > 0.95 (gym-pusht is_success)",
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Mean max coverage:     {summary['mean_max_coverage'] * 100:.2f}% (std {summary['std_max_coverage'] * 100:.2f}%)")
    print(f"  Mean final coverage:   {summary['mean_final_coverage'] * 100:.2f}%")
    print(f"  Success rate (>95%):   {summary['success_rate'] * 100:.1f}%")
    print(f"  Max coverage >= 80%:   {summary['rate_max_cov_ge_0.80'] * 100:.1f}%")
    print(f"  Mean FDE (plan vs executed): {summary['mean_fde']:.4f}")

    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump({"settings": settings, "summary": summary, "episodes": results}, f, indent=2)

    # Hardest third of episodes, for inspection / targeted data collection
    hard_cases = sorted(results, key=lambda r: r["max_coverage"])[: max(2, len(results) // 3)]
    with open(os.path.join(args.out_dir, "hard_cases.json"), "w") as f:
        json.dump({"settings": settings, "summary": summary, "hard_cases": hard_cases}, f, indent=2)

    print(f"  Results:  {os.path.join(args.out_dir, 'eval_results.json')}")
    print(f"  Hard cases: {os.path.join(args.out_dir, 'hard_cases.json')}")
    if recorder is not None:
        print(f"  Video:    {video_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

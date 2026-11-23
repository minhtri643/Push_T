"""Step 3: collect extra robot-object interaction data in the Push-T simulator.

Expert demonstrations alone teach the World Model little about "what if the robot pushed
differently": in expert data the next action is almost a function of the state, so a model can
ignore the action and still fit. This script replays every TRAIN demonstration from its real
initial state with smooth random perturbations of the actions, plus short random pokes near the
block, and records the resulting physics. Only train episodes are used (no validation leakage).

Output: npz with concatenated arrays agent (N,2), block (N,3), action (N,2) in pixels and
per-episode `lengths`; read it with data.dynamics_data.load_sim_sequences.

    python experiments/E3_dynamics_world_model/code/collect_sim_interactions.py --variants 3
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import time

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import yaml

from pusht_wm_dit.data.dynamics_data import load_demo_sequences

NOISE_PX = (10.0, 20.0, 35.0)  # std of the smooth action perturbation, one per variant


def smooth_noise(rng, T, sigma, rho=0.85):
    n = np.zeros((T, 2))
    for t in range(1, T):
        n[t] = rho * n[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma, 2)
    return n


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=int, default=3, help="perturbed replays per train episode")
    p.add_argument("--poke-prob", type=float, default=0.03, help="per-step chance to start a random poke")
    p.add_argument("--out", default=cfg["dynamics"]["sim_data_path"])
    p.add_argument("--images", action="store_true", help="also save the 96x96 frame of every step to <out>.frames.npy (for the vision World Model)")
    args = p.parse_args()

    seqs = load_demo_sequences(cfg["data"]["zarr_path"], "train", cfg["data"]["val_ratio"], cfg["seed"])
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos" if args.images else "state")
    frames = []
    env.reset(seed=cfg["seed"])
    u = env.unwrapped
    rng = np.random.default_rng(cfg["seed"])
    agents, blocks, actions, lengths = [], [], [], []
    t0 = time.perf_counter()
    for i, s in enumerate(seqs):
        for v in range(args.variants):
            env.reset(seed=int(rng.integers(1 << 30)))
            st = list(s["agent"][0]) + list(s["block"][0])
            u._set_state(st)
            u._set_state(st)  # twice: exact placement (pymunk rotates about the centre of gravity)
            T = len(s["action"])
            noise = smooth_noise(rng, T, NOISE_PX[v % len(NOISE_PX)])
            ag, bl, ac = [], [], []
            poke_left, poke_target = 0, None
            for t in range(T):
                info = u._get_info()
                a = s["action"][t] + noise[t]
                if poke_left == 0 and rng.random() < args.poke_prob:
                    bx, by, th = info["block_pose"]
                    r, phi = rng.uniform(20, 90), rng.uniform(-np.pi, np.pi)
                    poke_target = np.array([bx + r * np.cos(phi), by + 45 + r * np.sin(phi)])
                    poke_left = int(rng.integers(4, 12))
                if poke_left > 0:
                    a = poke_target
                    poke_left -= 1
                a = np.clip(a, 10.0, 502.0)
                if args.images:
                    frames.append(u.get_obs()["pixels"])  # frame at time t, before the action
                ag.append(np.array(info["pos_agent"], dtype=np.float64))
                bl.append(np.array(info["block_pose"], dtype=np.float64))
                ac.append(a)
                u.step(a)
            agents.append(np.array(ag))
            blocks.append(np.array(bl))
            actions.append(np.array(ac))
            lengths.append(T)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(seqs)} episodes ({sum(lengths):,} steps, {time.perf_counter() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, agent=np.concatenate(agents), block=np.concatenate(blocks),
                        action=np.concatenate(actions), lengths=np.array(lengths))
    print(f"Saved {sum(lengths):,} simulated steps from {len(lengths)} rollouts to {args.out}")
    if args.images:
        np.save(args.out + ".frames.npy", np.stack(frames).astype(np.uint8))
        print(f"Saved {len(frames):,} frames to {args.out}.frames.npy")


if __name__ == "__main__":
    main()

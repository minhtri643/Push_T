"""Comprehensive Systematic Evaluation Across State Space.

Tests the World Model + DiT + Inverse Dynamics policy across 20 diverse
initial conditions spanning all 4 quadrants, extreme angles, and wall proximities.
Categorizes failure modes to isolate out-of-distribution weaknesses.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import math
import numpy as np
import torch
import yaml
from typing import Dict, List, Any

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.models.world_model import WorldModel
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    four_d_to_pose,
)
from pusht_wm_dit.utils.metrics import compute_fde_numpy


def evaluate_custom_state(
    sim: PushTSim,
    wm: WorldModel,
    dit: DiTTrajectoryPlanner,
    id_ctrl: InverseDynamicsController,
    device: str,
    block_x: float,
    block_y: float,
    block_th: float,
    agent_x: float,
    agent_y: float,
    max_steps: int = 180,
    exec_horizon: int = 8,
    sample_steps: int = 10,
) -> Dict[str, Any]:
    """Runs rollout from a specific arbitrary (x, y, theta) placement."""
    sim.reset()
    state = [agent_x, agent_y, block_x, block_y, block_th]
    sim.env.unwrapped._set_state(state)
    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

    max_cov = obs["coverage"]
    init_cov = obs["coverage"]
    init_dist = math.hypot(block_x - 256.0, block_y - 256.0)

    step = 0
    terminated = False
    fde_history = []
    goal_dist_history = [init_dist]

    while step < max_steps and not terminated:
        frame_t = torch.from_numpy(normalize_image(obs["pixels"])).float().unsqueeze(0).to(device)
        agent_t = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
        curr_pose_t = torch.from_numpy(obs["block_pose_4d"]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            s_t = wm.encode(frame_t, agent_t)
            hat_path = dit.sample(
                s_t,
                num_inference_steps=sample_steps,
                current_pose_4d=curr_pose_t,
                best_of_n=8,
            )
            pred_actions = id_ctrl(s_t, hat_path, agent_pos=agent_t)[0].cpu().numpy()

        hat_path_np = hat_path[0].cpu().numpy()

        for a_idx in range(min(exec_horizon, len(pred_actions))):
            act_step = pred_actions[a_idx]
            act_px = unnormalize_pos(act_step)
            curr_agent_px = obs["agent_pos_px"]
            delta = act_px - curr_agent_px
            dist = np.linalg.norm(delta)
            if dist > 40.0:
                act_px = curr_agent_px + (delta / dist) * 40.0
                act_step = normalize_pos(act_px)

            obs, r, term, trunc, info = sim.step(act_step, normalized_action=True)
            step += 1

            cov = obs["coverage"]
            if cov > max_cov:
                max_cov = cov

            bx, by = obs["block_pose"][0], obs["block_pose"][1]
            curr_dist = math.hypot(bx - 256.0, by - 256.0)
            goal_dist_history.append(curr_dist)

            step_fde = compute_fde_numpy(obs["block_pose_4d"], hat_path_np[min(a_idx, len(hat_path_np) - 1)])
            fde_history.append(step_fde)

            # Target Lock
            if cov >= 0.82:
                terminated = True
                break
            if max_cov >= 0.72 and cov < max_cov - 0.04:
                terminated = True
                break

    final_dist = goal_dist_history[-1]
    final_cov = obs["coverage"]
    final_pose = obs["block_pose"].tolist()

    # Diagnose failure mode
    if max_cov >= 0.75:
        verdict = "SUCCESS_PLACED"
    elif final_dist < init_dist * 0.6:
        verdict = "APPROACHED_BUT_ANGLE_UNALIGNED"
    elif final_dist > init_dist + 20:
        verdict = "PUSHED_AWAY_FROM_GOAL"
    elif max(fde_history) > 0.40:
        verdict = "PUSHER_MISSED_OR_CIRCLED_INTO_WALL"
    else:
        verdict = "STUCK_NEAR_START"

    return {
        "init_block": [block_x, block_y, block_th],
        "init_agent": [agent_x, agent_y],
        "final_block": final_pose,
        "init_cov": float(init_cov),
        "max_cov": float(max_cov),
        "final_cov": float(final_cov),
        "init_dist": float(init_dist),
        "final_dist": float(final_dist),
        "mean_fde": float(np.mean(fde_history)) if fde_history else 0.0,
        "steps": step,
        "verdict": verdict,
    }


def main():
    config_path = "pusht_wm_dit/configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    wm_config = config["world_model"]
    dit_config = config["dit_planner"]
    id_config = config["inverse_dynamics"]

    print("=" * 70)
    print("RUNNING COMPREHENSIVE BENCHMARK SWEEP (20 SCENARIOS)")
    print(f"DEVICE: {device.upper()}")
    print("=" * 70)

    wm = WorldModel(
        latent_dim=wm_config["latent_dim"],
        agent_pos_dim=wm_config["agent_pos_dim"],
        pose_dim=wm_config["pose_dim"],
        action_dim=wm_config["action_dim"],
        num_groups=wm_config["num_groups"],
    ).to(device)
    wm.load_state_dict(torch.load(wm_config["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    wm.freeze()

    dit = DiTTrajectoryPlanner(
        horizon=dit_config["horizon"],
        traj_dim=dit_config["traj_dim"],
        latent_dim=wm_config["latent_dim"],
        model_dim=dit_config["model_dim"],
        num_heads=dit_config["num_heads"],
        num_layers=dit_config["num_layers"],
        mlp_ratio=dit_config["mlp_ratio"],
        diffusion_steps=dit_config["diffusion_steps"],
        beta_schedule=dit_config["beta_schedule"],
        w_xy=dit_config["w_xy"],
        w_theta=dit_config["w_theta"],
        lambda_fde=dit_config["lambda_fde"],
    ).to(device)
    dit.load_state_dict(torch.load(dit_config["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    dit.eval()

    id_ctrl = InverseDynamicsController(
        latent_dim=wm_config["latent_dim"],
        horizon=dit_config["horizon"],
        traj_dim=dit_config["traj_dim"],
        action_dim=id_config["action_dim"],
        exec_steps=id_config["exec_steps"],
        hidden_dims=tuple(id_config["hidden_dims"]),
    ).to(device)
    id_ctrl.load_state_dict(torch.load(id_config["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    id_ctrl.eval()

    sim = PushTSim(env_name="gym_pusht/PushT-v0", render_size=(96, 96), max_steps=200)

    # Define 20 test cases spanning the entire 512x512 space:
    test_cases = [
        # 1. Central Near-Goal approaches
        {"name": "Center-Bottom", "bx": 256, "by": 340, "th": math.pi / 4, "ax": 256, "ay": 420},
        {"name": "Center-Right", "bx": 340, "by": 256, "th": math.pi / 4, "ax": 420, "ay": 256},
        {"name": "Center-Top", "bx": 256, "by": 170, "th": math.pi / 4, "ax": 256, "ay": 100},
        {"name": "Center-Left", "bx": 170, "by": 256, "th": math.pi / 4, "ax": 100, "ay": 256},

        # 2. Quadrant 1 (Top-Right)
        {"name": "Q1 Top-Right (Normal angle)", "bx": 360, "by": 160, "th": 0.5, "ax": 400, "ay": 120},
        {"name": "Q1 Top-Right (Inverted angle)", "bx": 360, "by": 160, "th": -math.pi / 2, "ax": 320, "ay": 200},
        {"name": "Q1 Near Top Wall", "bx": 320, "by": 90, "th": 0.0, "ax": 320, "ay": 40},

        # 3. Quadrant 2 (Top-Left)
        {"name": "Q2 Top-Left (Standard)", "bx": 150, "by": 160, "th": math.pi / 4, "ax": 100, "ay": 120},
        {"name": "Q2 Top-Left (Opposite angle)", "bx": 150, "by": 160, "th": -2.0, "ax": 180, "ay": 200},
        {"name": "Q2 Near Left Wall", "bx": 90, "by": 200, "th": 1.0, "ax": 40, "ay": 200},

        # 4. Quadrant 3 (Bottom-Left)
        {"name": "Q3 Bottom-Left (Standard)", "bx": 160, "by": 360, "th": math.pi / 4, "ax": 120, "ay": 420},
        {"name": "Q3 Bottom-Left (Extreme Angle)", "bx": 160, "by": 360, "th": math.pi, "ax": 200, "ay": 320},
        {"name": "Q3 Near Corner (80, 420)", "bx": 100, "by": 420, "th": -0.5, "ax": 60, "ay": 450},

        # 5. Quadrant 4 (Bottom-Right)
        {"name": "Q4 Bottom-Right (Standard)", "bx": 360, "by": 360, "th": math.pi / 4, "ax": 420, "ay": 420},
        {"name": "Q4 Bottom-Right (Rotated 90 deg)", "bx": 360, "by": 360, "th": -math.pi / 4, "ax": 320, "ay": 400},
        {"name": "Q4 Near Right Wall", "bx": 430, "by": 320, "th": 0.0, "ax": 470, "ay": 320},

        # 6. Pusher starting far away / Opposite side
        {"name": "Pusher Opposite Side (Block Q3, Agent Q1)", "bx": 180, "by": 350, "th": 0.78, "ax": 400, "ay": 120},
        {"name": "Pusher Trapped Behind Goal", "bx": 280, "by": 300, "th": 0.78, "ax": 200, "ay": 200},
        {"name": "Inverted Block at Center", "bx": 256, "by": 256, "th": -math.pi * 0.75, "ax": 320, "ay": 320},
        {"name": "Block Tilted 180 at Center", "bx": 256, "by": 256, "th": math.pi + 0.78, "ax": 180, "ay": 180},
    ]

    results = []
    print(f"\nEvaluating across {len(test_cases)} diverse initial configurations...\n")

    for idx, tc in enumerate(test_cases):
        res = evaluate_custom_state(
            sim=sim,
            wm=wm,
            dit=dit,
            id_ctrl=id_ctrl,
            device=device,
            block_x=tc["bx"],
            block_y=tc["by"],
            block_th=tc["th"],
            agent_x=tc["ax"],
            agent_y=tc["ay"],
        )
        res["test_name"] = tc["name"]
        results.append(res)
        print(
            f"[{idx+1:02d}/20] {tc['name']:<35} | "
            f"Max Cov: {res['max_cov']*100:5.1f}% | "
            f"Final Dist: {res['final_dist']:5.1f}px (from {res['init_dist']:5.1f}px) | "
            f"Verdict: {res['verdict']}"
        )

    sim.close()

    # Aggregate Analysis
    verdicts = [r["verdict"] for r in results]
    success_count = sum(1 for v in verdicts if v == "SUCCESS_PLACED")
    angle_fail_count = sum(1 for v in verdicts if v == "APPROACHED_BUT_ANGLE_UNALIGNED")
    push_away_count = sum(1 for v in verdicts if v == "PUSHED_AWAY_FROM_GOAL")
    wall_miss_count = sum(1 for v in verdicts if v in ["PUSHER_MISSED_OR_CIRCLED_INTO_WALL", "STUCK_NEAR_START"])

    summary = {
        "total_tests": len(test_cases),
        "success_placed": success_count,
        "approached_but_angle_unaligned": angle_fail_count,
        "pushed_away_from_goal": push_away_count,
        "wall_or_pusher_missed": wall_miss_count,
        "success_rate": float(success_count / len(test_cases) * 100),
        "mean_max_coverage": float(np.mean([r["max_cov"] for r in results]) * 100),
        "mean_final_distance": float(np.mean([r["final_dist"] for r in results])),
        "detailed_results": results,
    }

    out_path = "pusht_wm_dit/eval_videos/comprehensive_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY:")
    print("=" * 70)
    print(f"  Total Scenarios Tested:              {len(test_cases)}")
    print(f"  Successfully Placed (IoU >= 75%):    {success_count} ({success_count/len(test_cases)*100:.1f}%)")
    print(f"  Approached Goal, But Angle Unaligned: {angle_fail_count} ({angle_fail_count/len(test_cases)*100:.1f}%)")
    print(f"  Pushed Away from Goal:               {push_away_count} ({push_away_count/len(test_cases)*100:.1f}%)")
    print(f"  Missed Contact / Stuck Near Wall:    {wall_miss_count} ({wall_miss_count/len(test_cases)*100:.1f}%)")
    print(f"  Mean Peak Coverage:                  {summary['mean_max_coverage']:.2f}%")
    print(f"  Results saved to:                    {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

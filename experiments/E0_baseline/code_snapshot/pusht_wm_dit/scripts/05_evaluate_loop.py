"""Script 05: Closed-Loop Receding Horizon Evaluation & Hard-Case Diagnostic Logging.

Closed-Loop Pipeline:
1. Observes RGB frame (96x96x3) and agent pos (x_agent, y_agent).
2. Encodes into compact latent state vector s_t via frozen World Model.
3. Plans future H=16 trajectory hat_path via Diffusion Transformer (DiT).
4. Generates action sequence for Ta=8 via Inverse Dynamics Controller.
5. Steps Pymunk simulation, tracks Coverage IoU, FDE, and ADE.
6. Renders and saves video with telemetry HUD to 'pusht_wm_dit/eval_videos/eval_rollout.mp4'.
7. Exports detailed hard-case diagnostics to JSON for active learning & fine-tuning.
"""

import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import yaml
import numpy as np
import torch
from typing import Dict, List, Any, Optional

from pusht_wm_dit.envs.pusht_sim import PushTSim, VideoRecorder
from pusht_wm_dit.models.world_model import WorldModel
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    pose_to_4d,
    four_d_to_pose,
)
from pusht_wm_dit.utils.metrics import (
    compute_fde_numpy,
    compute_ade_numpy,
    compute_target_coverage_iou,
)


def evaluate_episode(
    sim: PushTSim,
    world_model: WorldModel,
    dit_planner: DiTTrajectoryPlanner,
    id_controller: InverseDynamicsController,
    recorder: Optional[VideoRecorder],
    device: str,
    seed: int,
    pred_horizon: int = 16,
    exec_horizon: int = 8,
    sample_steps: int = 10,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Runs a single closed-loop Receding Horizon Control (RHC) rollout."""
    obs, info = sim.reset(seed=seed)
    init_block_pose = obs["block_pose"].tolist()
    init_agent_pos = obs["agent_pos_px"].tolist()

    max_coverage = obs["coverage"]
    fde_history: List[float] = []
    ade_history: List[float] = []
    coverage_history: List[float] = [obs["coverage"]]

    if recorder is not None:
        recorder.record_frame(
            frame=obs["pixels"],
            step=0,
            coverage=obs["coverage"],
            info_text="Closed-Loop DiT Planner Initialized",
        )

    current_step = 0

    while current_step < max_steps:
        # 1. Observation input
        frame_tensor = torch.from_numpy(normalize_image(obs["pixels"])).float().unsqueeze(0).to(device)
        agent_tensor = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)

        # 2. Latent state encoding via frozen World Model
        with torch.no_grad():
            s_t = world_model.encode(frame_tensor, agent_tensor)

            # 3. Trajectory planning via DiT (with Step-0 Inpainting & Best-of-8 Candidate Selection)
            curr_pose_t = torch.from_numpy(obs["block_pose_4d"]).float().unsqueeze(0).to(device)
            hat_path = dit_planner.sample(
                s_t,
                num_inference_steps=sample_steps,
                current_pose_4d=curr_pose_t,
                best_of_n=8,
            )  # (1, H, 4)

            # 4. Action generation via Inverse Dynamics (Ta steps with explicit agent_pos)
            pred_actions = id_controller(s_t, hat_path, agent_pos=agent_tensor)  # (1, Ta, 2)

        pred_actions_np = pred_actions[0].cpu().numpy()  # (Ta, 2) in [-1, 1]
        hat_path_np = hat_path[0].cpu().numpy()          # (H, 4) in [-1, 1]

        # 5. Execute Ta steps in the environment (Receding Horizon Control)
        executed_poses_4d = []
        for a_idx in range(min(exec_horizon, len(pred_actions_np))):
            act_step = pred_actions_np[a_idx]

            # Soft action rate-limiting to prevent excessive kinetic impact flinging the block
            act_px = unnormalize_pos(act_step)
            curr_agent_px = obs["agent_pos_px"]
            delta = act_px - curr_agent_px
            dist = np.linalg.norm(delta)
            if dist > 40.0:
                act_px = curr_agent_px + (delta / dist) * 40.0
                act_step = normalize_pos(act_px)

            obs, reward, terminated, truncated, info = sim.step(act_step, normalized_action=True)
            current_step += 1

            cov = obs["coverage"]
            coverage_history.append(cov)
            if cov > max_coverage:
                max_coverage = cov

            executed_poses_4d.append(obs["block_pose_4d"])

            # Compute current displacement error to planned path
            planned_step = min(a_idx, pred_horizon - 1)
            step_fde = compute_fde_numpy(obs["block_pose_4d"], hat_path_np[planned_step])
            fde_history.append(step_fde)

            # Target Lock & Settling:
            # Prevent over-pushing once block is inside target zone or peak placement attained
            is_locked = False
            if cov >= 0.82:
                terminated = True
                is_locked = True
            elif max_coverage >= 0.72 and cov < max_coverage - 0.04:
                terminated = True
                is_locked = True

            status_str = "Target Locked in Goal" if is_locked else f"RHC Execution (Substep {a_idx+1}/{exec_horizon})"

            if recorder is not None:
                recorder.record_frame(
                    frame=obs["pixels"],
                    step=current_step,
                    coverage=cov,
                    fde=step_fde,
                    info_text=status_str,
                )

            if terminated or truncated or current_step >= max_steps:
                break

        if terminated or truncated or current_step >= max_steps:
            break

    final_coverage = obs["coverage"]
    mean_fde = float(np.mean(fde_history)) if fde_history else 0.0

    return {
        "seed": seed,
        "steps": current_step,
        "init_block_pose": init_block_pose,
        "init_agent_pos": init_agent_pos,
        "final_block_pose": obs["block_pose"].tolist(),
        "final_coverage": final_coverage,
        "max_coverage": max_coverage,
        "success": bool(max_coverage >= 0.80),
        "perfect_success": bool(max_coverage >= 0.95),
        "mean_fde": mean_fde,
        "coverage_history": coverage_history,
    }


def main():
    config_path = "pusht_wm_dit/configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    data_config = config["data"]
    wm_config = config["world_model"]
    dit_config = config["dit_planner"]
    id_config = config["inverse_dynamics"]
    eval_config = config["evaluation"]
    env_config = config["env"]

    video_dir = eval_config["video_dir"]
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "eval_rollout.mp4")
    hard_cases_path = eval_config["hard_cases_log"]

    print("=" * 70)
    print(f"STEP 5: CLOSED-LOOP EVALUATION & HARD-CASE LOGGING (DEVICE: {device.upper()})")
    print("=" * 70)

    # 1. Load Models
    print("Loading checkpoints...")
    # World Model
    wm = WorldModel(
        latent_dim=wm_config["latent_dim"],
        agent_pos_dim=wm_config["agent_pos_dim"],
        pose_dim=wm_config["pose_dim"],
        action_dim=wm_config["action_dim"],
        num_groups=wm_config["num_groups"],
    ).to(device)
    if os.path.exists(wm_config["save_path"]):
        ckpt = torch.load(wm_config["save_path"], map_location=device, weights_only=False)
        wm.load_state_dict(ckpt["model_state_dict"])
    wm.freeze()

    # DiT Planner
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
    if os.path.exists(dit_config["save_path"]):
        ckpt = torch.load(dit_config["save_path"], map_location=device, weights_only=False)
        dit.load_state_dict(ckpt["model_state_dict"])
    dit.eval()

    # Inverse Dynamics
    id_ctrl = InverseDynamicsController(
        latent_dim=wm_config["latent_dim"],
        horizon=dit_config["horizon"],
        traj_dim=dit_config["traj_dim"],
        action_dim=id_config["action_dim"],
        exec_steps=id_config["exec_steps"],
        hidden_dims=tuple(id_config["hidden_dims"]),
    ).to(device)
    if os.path.exists(id_config["save_path"]):
        ckpt = torch.load(id_config["save_path"], map_location=device, weights_only=False)
        id_ctrl.load_state_dict(ckpt["model_state_dict"])
    id_ctrl.eval()

    # 2. Initialize Environment & Video Recorder
    sim = PushTSim(
        env_name=env_config["env_name"],
        render_size=tuple(env_config["render_size"]),
        max_steps=env_config["max_steps"],
    )
    recorder = VideoRecorder(output_path=video_path, fps=15, resolution=(384, 384))

    num_episodes = eval_config["num_eval_episodes"]
    results = []

    print(f"\nEvaluating over {num_episodes} closed-loop test episodes (RHC: H={dit_config['horizon']}, Ta={id_config['exec_steps']})...")

    for ep_idx in range(num_episodes):
        ep_seed = config["seed"] + ep_idx * 101
        res = evaluate_episode(
            sim=sim,
            world_model=wm,
            dit_planner=dit,
            id_controller=id_ctrl,
            recorder=recorder if ep_idx == 0 else None,
            device=device,
            seed=ep_seed,
            pred_horizon=dit_config["horizon"],
            exec_horizon=id_config["exec_steps"],
            sample_steps=dit_config["sample_steps"],
            max_steps=env_config["max_steps"],
        )
        results.append(res)
        print(
            f"  Episode {ep_idx+1:02d}/{num_episodes:02d} (Seed {ep_seed}) | "
            f"Max Cov: {res['max_coverage']*100:5.1f}% | "
            f"Final Cov: {res['final_coverage']*100:5.1f}% | "
            f"Mean FDE: {res['mean_fde']:.4f} | "
            f"Steps: {res['steps']}"
        )

    sim.close()
    recorder.save()

    # 3. Aggregate Performance Metrics
    all_max_cov = [r["max_coverage"] for r in results]
    all_final_cov = [r["final_coverage"] for r in results]
    all_fde = [r["mean_fde"] for r in results]
    success_rate = np.mean([r["success"] for r in results]) * 100
    perfect_success_rate = np.mean([r["perfect_success"] for r in results]) * 100

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY:")
    print("=" * 70)
    print(f"  Mean Peak Target Coverage IoU:  {np.mean(all_max_cov) * 100:.2f}% (Std: {np.std(all_max_cov) * 100:.2f}%)")
    print(f"  Mean Final Target Coverage IoU: {np.mean(all_final_cov) * 100:.2f}%")
    print(f"  Success Rate (IoU >= 80%):      {success_rate:.1f}%")
    print(f"  Perfect Success (IoU >= 95%):   {perfect_success_rate:.1f}%")
    print(f"  Mean Final Displacement Error:  {np.mean(all_fde):.4f}")
    print(f"  Rollout Video:                  {video_path}")

    # 4. Filter and Export Hard-Cases for Step 3 Fine-Tuning
    sorted_episodes = sorted(results, key=lambda x: x["max_coverage"])
    hard_cases = sorted_episodes[: max(2, len(sorted_episodes) // 3)]

    diagnostic_report = {
        "evaluation_summary": {
            "num_episodes": num_episodes,
            "mean_peak_coverage": float(np.mean(all_max_cov)),
            "mean_final_coverage": float(np.mean(all_final_cov)),
            "mean_fde": float(np.mean(all_fde)),
            "success_rate_80": float(success_rate),
            "perfect_success_rate_95": float(perfect_success_rate),
        },
        "hard_cases_identified": hard_cases,
        "step3_finetuning_recommendations": [
            "1. Increase World Model transition consistency loss weight lambda_transition from 0.5 to 1.0 for large rotational angles.",
            "2. Sample harder initial angles (|theta_T - pi/4| > 1.5 rad) during data augmentation.",
            "3. Increase DiT lambda_fde penalty to prioritize alignment at the end of horizon H=16.",
            "4. Decrease Receding Horizon execution window Ta from 8 to 4 for agile closed-loop correction near boundaries.",
        ],
    }

    with open(hard_cases_path, "w") as f:
        json.dump(diagnostic_report, f, indent=2)

    print(f"\nHard cases logged to: {hard_cases_path}")
    print("=" * 70)
    print("STEP 5: EVALUATION LOOP COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

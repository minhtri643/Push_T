"""Random User Benchmark Simulation.

Simulates a human user interacting with the interactive GUI, placing the T-block
and the pusher circle at arbitrary random positions and orientations across the board.
Evaluates the system's robustness, user experience, and failure modes.
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


def run_random_trial(
    trial_idx: int,
    sim: PushTSim,
    wm: WorldModel,
    dit: DiTTrajectoryPlanner,
    id_ctrl: InverseDynamicsController,
    device: str,
    bx: float,
    by: float,
    bth: float,
    ax: float,
    ay: float,
    max_steps: int = 240,
    exec_horizon: int = 8,
    sample_steps: int = 10,
) -> Dict[str, Any]:
    """Runs a single trial mimicking a user randomly placing T and pusher."""
    sim.reset()
    state = [ax, ay, bx, by, bth]
    sim.env.unwrapped._set_state(state)
    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

    init_cov = obs["coverage"]
    init_dist = math.hypot(bx - 256.0, by - 256.0)
    init_dth = abs((bth - math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi)

    max_cov = init_cov
    step_count = 0
    pending_actions = []
    goal_dist_history = [init_dist]
    cov_history = [init_cov]
    terminated = False
    status_reason = "MAX_STEPS"

    while step_count < max_steps and not terminated:
        if len(pending_actions) == 0:
            frame_tensor = torch.from_numpy(normalize_image(obs["pixels"])).float().unsqueeze(0).to(device)
            agent_tensor = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
            curr_pose_t = torch.from_numpy(obs["block_pose_4d"]).float().unsqueeze(0).to(device)

            with torch.no_grad():
                s_t = wm.encode(frame_tensor, agent_tensor)
                hat_path = dit.sample(
                    s_t,
                    num_inference_steps=sample_steps,
                    current_pose_4d=curr_pose_t,
                    best_of_n=8,
                )
                actions = id_ctrl(s_t, hat_path, agent_pos=agent_tensor)

            pred_acts = actions[0].cpu().numpy()
            for a_idx in range(min(exec_horizon, len(pred_acts))):
                pending_actions.append(pred_acts[a_idx])

        if len(pending_actions) > 0:
            act_step = pending_actions.pop(0)

            # Smart Stance Navigation
            curr_bx, curr_by = obs["block_pose"][0], obs["block_pose"][1]
            curr_ax, curr_ay = obs["agent_pos_px"][0], obs["agent_pos_px"][1]
            gx, gy = 256.0, 256.0
            dist_g = math.hypot(gx - curr_bx, gy - curr_by)

            if dist_g > 35.0:
                ux = (gx - curr_bx) / (dist_g + 1e-6)
                uy = (gy - curr_by) / (dist_g + 1e-6)
                rx = curr_ax - curr_bx
                ry = curr_ay - curr_by
                alignment = rx * ux + ry * uy
                if alignment > 20.0:  # Agent is between block and goal!
                    perp_x = -uy
                    perp_y = ux
                    side = 1.0 if (rx * perp_x + ry * perp_y) >= 0 else -1.0
                    wp_x = curr_bx + side * perp_x * 75.0 - ux * 25.0
                    wp_y = curr_by + side * perp_y * 75.0 - uy * 25.0
                    act_step = normalize_pos(np.array([wp_x, wp_y], dtype=np.float32))

            # Rate limiting
            act_px = unnormalize_pos(act_step)
            curr_agent_px = obs["agent_pos_px"]
            delta = act_px - curr_agent_px
            dist = np.linalg.norm(delta)
            if dist > 40.0:
                act_px = curr_agent_px + (delta / dist) * 40.0
                act_step = normalize_pos(act_px)

            obs, reward, term, trunc, info = sim.step(act_step, normalized_action=True)
            step_count += 1
            cov = obs["coverage"]
            cov_history.append(cov)
            if cov > max_cov:
                max_cov = cov

            cur_dist = math.hypot(obs["block_pose"][0] - 256.0, obs["block_pose"][1] - 256.0)
            goal_dist_history.append(cur_dist)

            # Target Lock
            if cov >= 0.82:
                terminated = True
                status_reason = "GOAL_REACHED"
                break
            elif max_cov >= 0.72 and cov < max_cov - 0.04:
                terminated = True
                status_reason = "TARGET_LOCKED"
                break

    final_dist = goal_dist_history[-1]
    final_cov = cov_history[-1]
    final_bth = obs["block_pose"][2]
    final_dth = abs((final_bth - math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi)

    # Classify user-experience verdict
    if max_cov >= 0.78:
        verdict = "HOÀN HẢO (Vào ô chuẩn)"
    elif max_cov >= 0.65:
        verdict = "KHÁ TỐT (Đúng ô, hơi lệch góc)"
    elif final_dist < init_dist * 0.5:
        verdict = "ĐẨY GẦN ĐÍCH (Chưa xoay khớp ô)"
    elif final_dist > init_dist + 20.0:
        verdict = "ĐẨY CHỆCH HƯỚNG (Đi xa đích hơn)"
    else:
        verdict = "KẸT / KHÔNG TIẾP XÚC HIỆU QUẢ"

    return {
        "trial": trial_idx,
        "init_block": [round(bx, 1), round(by, 1), round(bth, 2)],
        "init_agent": [round(ax, 1), round(ay, 1)],
        "init_dist": round(init_dist, 1),
        "final_dist": round(final_dist, 1),
        "init_cov": round(init_cov * 100, 1),
        "max_cov": round(max_cov * 100, 1),
        "final_cov": round(final_cov * 100, 1),
        "init_dth_deg": round(math.degrees(init_dth), 1),
        "final_dth_deg": round(math.degrees(final_dth), 1),
        "steps": step_count,
        "status_reason": status_reason,
        "verdict": verdict,
    }


def main():
    config_path = "pusht_wm_dit/configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"CHƠI THỬ & ĐÁNH GIÁ NGẪU NHIÊN TỪ GÓC NHÌN NGƯỜI DÙNG (DEVICE: {device.upper()})")
    print("=" * 75)

    wm_cfg = config["world_model"]
    wm = WorldModel(
        latent_dim=wm_cfg["latent_dim"],
        agent_pos_dim=wm_cfg["agent_pos_dim"],
        pose_dim=wm_cfg["pose_dim"],
        action_dim=wm_cfg["action_dim"],
        num_groups=wm_cfg["num_groups"],
    ).to(device)
    wm.load_state_dict(torch.load(wm_cfg["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    wm.eval()

    dit_cfg = config["dit_planner"]
    dit = DiTTrajectoryPlanner(
        horizon=dit_cfg["horizon"],
        traj_dim=dit_cfg["traj_dim"],
        latent_dim=wm_cfg["latent_dim"],
        model_dim=dit_cfg["model_dim"],
        num_heads=dit_cfg["num_heads"],
        num_layers=dit_cfg["num_layers"],
    ).to(device)
    dit.load_state_dict(torch.load(dit_cfg["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    dit.eval()

    id_cfg = config["inverse_dynamics"]
    id_ctrl = InverseDynamicsController(
        latent_dim=wm_cfg["latent_dim"],
        horizon=dit_cfg["horizon"],
        traj_dim=dit_cfg["traj_dim"],
        action_dim=id_cfg["action_dim"],
        exec_steps=id_cfg["exec_steps"],
    ).to(device)
    id_ctrl.load_state_dict(torch.load(id_cfg["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    id_ctrl.eval()

    sim = PushTSim(render_size=(96, 96))

    # Seed for reproducible random user placements
    np.random.seed(2026)
    num_trials = 25
    results = []

    print(f"\nTiến hành thử nghiệm {num_trials} lượt đặt ngẫu nhiên của người dùng...")
    print(f"{'Lượt':<4} | {'Vị trí T ban đầu':<18} | {'Khoảng cách':<16} | {'Max IoU':<8} | {'Bước':<5} | {'Đánh giá trải nghiệm'}")
    print("-" * 88)

    for i in range(1, num_trials + 1):
        # Human player places T somewhere in safe interior [100, 412]
        bx = float(np.random.uniform(110.0, 400.0))
        by = float(np.random.uniform(110.0, 400.0))
        # Random arbitrary orientation [-pi, pi]
        bth = float(np.random.uniform(-math.pi, math.pi))
        # Human player places pusher circle nearby or anywhere
        ax = float(np.random.uniform(90.0, 420.0))
        ay = float(np.random.uniform(90.0, 420.0))

        res = run_random_trial(
            trial_idx=i,
            sim=sim,
            wm=wm,
            dit=dit,
            id_ctrl=id_ctrl,
            device=device,
            bx=bx,
            by=by,
            bth=bth,
            ax=ax,
            ay=ay,
        )
        results.append(res)

        pos_str = f"({res['init_block'][0]:.0f}, {res['init_block'][1]:.0f})"
        dist_str = f"{res['init_dist']:5.1f} -> {res['final_dist']:5.1f}px"
        cov_str = f"{res['max_cov']:5.1f}%"
        print(f"#{i:02d}  | {pos_str:<18} | {dist_str:<16} | {cov_str:<8} | {res['steps']:<5} | {res['verdict']}")

    # Aggregating results
    covs = [r["max_cov"] for r in results]
    success_high = sum(1 for r in results if r["max_cov"] >= 78.0)
    success_mid = sum(1 for r in results if r["max_cov"] >= 65.0)
    approached = sum(1 for r in results if r["final_dist"] < r["init_dist"] * 0.6)
    pushed_away = sum(1 for r in results if r["final_dist"] > r["init_dist"] + 20.0)

    print("\n" + "=" * 75)
    print("TỔNG HỢP ĐÁNH GIÁ TRẢI NGHIỆM NGƯỜI DÙNG ĐẶT RANDOM:")
    print("=" * 75)
    print(f"  Tổng số lượt thử:                    {num_trials}")
    print(f"  Thành công hoàn hảo (IoU >= 78%):    {success_high}/{num_trials} ({success_high/num_trials*100:.1f}%)")
    print(f"  Vào ô khá chuẩn (IoU >= 65%):        {success_mid}/{num_trials} ({success_mid/num_trials*100:.1f}%)")
    print(f"  Kéo vật thể lại gần đích (< 60% d0): {approached}/{num_trials} ({approached/num_trials*100:.1f}%)")
    print(f"  Đẩy chệch hướng / Văng xa đích:      {pushed_away}/{num_trials} ({pushed_away/num_trials*100:.1f}%)")
    print(f"  Độ phủ tối đa trung bình (Mean IoU): {np.mean(covs):.1f}% (Max: {np.max(covs):.1f}%)")

    # Save to JSON
    out_path = "pusht_wm_dit/eval_videos/random_user_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_trials": num_trials,
                "mean_max_coverage": float(np.mean(covs)),
                "success_high_rate": float(success_high / num_trials),
                "success_mid_rate": float(success_mid / num_trials),
                "approached_rate": float(approached / num_trials),
                "pushed_away_rate": float(pushed_away / num_trials),
                "trials": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Chi tiết lưu tại: {out_path}")


if __name__ == "__main__":
    main()

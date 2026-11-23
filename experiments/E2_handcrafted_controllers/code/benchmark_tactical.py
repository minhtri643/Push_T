"""Benchmark comparing the Baseline Policy vs the User's 3-Phase Tactical Strategy.

Runs across identical random user placements (inverted angles, out-of-distribution positions).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import json
import math
import numpy as np
import torch
import yaml
from typing import Dict, List, Any

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.policy import build_policy
from controllers.tactical_controller import TacticalHierarchicalController
from pusht_wm_dit.utils.normalizer import normalize_pos


def run_single_eval(
    trial_idx: int,
    sim: PushTSim,
    policy: Any,
    controller: TacticalHierarchicalController,
    bx: float,
    by: float,
    bth: float,
    ax: float,
    ay: float,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    controller.reset()
    sim.reset()
    sim.env.unwrapped._set_state([ax, ay, bx, by, bth])
    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

    init_cov = obs["coverage"]
    init_dist = math.hypot(bx - 256.0, by - 256.0)
    init_dth = abs((bth - math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi)

    max_cov = init_cov
    step_count = 0
    phases_visited = set()

    for step in range(max_steps):
        act_px, status, info = controller.get_action(obs, sim, policy)
        cur_phase = info.get("phase", "UNKNOWN")
        phases_visited.add(cur_phase)

        # Rate limiting (max 35 px per step)
        curr_agent_px = obs["agent_pos_px"]
        delta = act_px - curr_agent_px
        dist = np.linalg.norm(delta)
        if dist > 35.0:
            act_px = curr_agent_px + (delta / dist) * 35.0

        act_norm = normalize_pos(act_px)
        obs, r, term, trunc, info_env = sim.step(act_norm, normalized_action=True)
        step_count += 1
        cov = obs["coverage"]
        if cov > max_cov:
            max_cov = cov

        if cur_phase == "LOCKED":
            break

    final_dist = math.hypot(obs["block_pose"][0] - 256.0, obs["block_pose"][1] - 256.0)
    final_dth = abs((obs["block_pose"][2] - math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi)

    if max_cov >= 0.78:
        verdict = "HOÀN HẢO (Vào ô chuẩn)"
    elif max_cov >= 0.55:
        verdict = "KHÁ TỐT (Đúng ô, lệch nhẹ)"
    elif final_dist < init_dist * 0.6:
        verdict = "ĐẨY GẦN ĐÍCH"
    elif final_dist > init_dist + 20.0:
        verdict = "ĐẨY CHỆCH HƯỚNG"
    else:
        verdict = "KẸT / CHƯA TỐI ƯU"

    return {
        "trial": trial_idx,
        "init_block": [round(bx, 1), round(by, 1), round(bth, 2)],
        "init_agent": [round(ax, 1), round(ay, 1)],
        "init_dist": round(init_dist, 1),
        "final_dist": round(final_dist, 1),
        "init_dth_deg": round(math.degrees(init_dth), 1),
        "final_dth_deg": round(math.degrees(final_dth), 1),
        "max_cov": round(max_cov * 100, 1),
        "final_cov": round(obs["coverage"] * 100, 1),
        "steps": step_count,
        "phases": list(phases_visited),
        "verdict": verdict,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"ĐO KIỂM CHIẾN THUẬT 3 NHỊP (ORIENT -> ALIGN -> PUSH) TRÊN DEVICE: {device.upper()}")
    print("=" * 80)

    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    policy = build_policy(config, device)
    controller = TacticalHierarchicalController()
    sim = PushTSim(render_size=(96, 96), max_steps=260)

    # Test across 15 reproducible random trials
    np.random.seed(2026)
    num_trials = 15
    results = []

    print(f"\n{'Lượt':<4} | {'Vị trí T ban đầu':<16} | {'Góc lệch':<10} | {'Khoảng cách':<16} | {'Max IoU':<8} | {'Bước':<5} | {'Đánh giá'}")
    print("-" * 90)

    for i in range(1, num_trials + 1):
        bx = float(np.random.uniform(120.0, 390.0))
        by = float(np.random.uniform(120.0, 390.0))
        bth = float(np.random.uniform(-math.pi, math.pi))
        ax = float(np.random.uniform(100.0, 410.0))
        ay = float(np.random.uniform(100.0, 410.0))

        res = run_single_eval(i, sim, policy, controller, bx, by, bth, ax, ay)
        results.append(res)

        pos_str = f"({res['init_block'][0]:.0f}, {res['init_block'][1]:.0f})"
        dth_str = f"{res['init_dth_deg']:4.0f}°->{res['final_dth_deg']:3.0f}°"
        dist_str = f"{res['init_dist']:5.1f} -> {res['final_dist']:5.1f}px"
        cov_str = f"{res['max_cov']:5.1f}%"
        print(f"#{i:02d}  | {pos_str:<16} | {dth_str:<10} | {dist_str:<16} | {cov_str:<8} | {res['steps']:<5} | {res['verdict']}")

    covs = [r["max_cov"] for r in results]
    success_high = sum(1 for r in results if r["max_cov"] >= 78.0)
    success_mid = sum(1 for r in results if r["max_cov"] >= 55.0)

    print("\n" + "=" * 80)
    print("TỔNG HỢP HIỆU QUẢ CHIẾN THUẬT:")
    print("=" * 80)
    print(f"  Thành công hoàn hảo (IoU >= 78%): {success_high}/{num_trials} ({success_high/num_trials*100:.1f}%)")
    print(f"  Vào ô khá chuẩn (IoU >= 55%):     {success_mid}/{num_trials} ({success_mid/num_trials*100:.1f}%)")
    print(f"  Độ phủ tối đa trung bình:         {np.mean(covs):.1f}% (Max: {np.max(covs):.1f}%)")


if __name__ == "__main__":
    main()

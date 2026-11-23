"""Play-test Suite: Evaluates and visualizes 5 distinct operational scenarios
to test model training effectiveness and control robustness.

Scenarios tested:
1. Standard Canonical: Moderate distance, small angle offset (~30 deg).
2. Extreme Inverted Angle: 160 deg rotation (OOD case for vanilla Diffusion).
3. Far Distance Quadrant: Distance > 200px across the workspace.
4. Near-Boundary Challenge: Positioned near wall edge.
5. Dynamic Moving Goal: Goal teleports halfway to test real-time target chaser.
"""

import os
import sys
import math
import json
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.policy import build_policy
from controllers.tactical_controller import TacticalHierarchicalController
from controllers.wm_lookahead import PushTCoverageEvaluator
from pusht_wm_dit.utils.normalizer import normalize_pos


def get_t_block_vertices(x: float, y: float, theta: float):
    """Returns the 8 vertices of the Push-T block in world coordinates."""
    # Local vertices matching Pymunk simulation
    local_pts = [
        (-100.0, -30.0), (100.0, -30.0), (100.0, 30.0), (30.0, 30.0),
        (30.0, 130.0), (-30.0, 130.0), (-30.0, 30.0), (-100.0, 30.0)
    ]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    world_pts = []
    for lx, ly in local_pts:
        wx = x + lx * cos_t - ly * sin_t
        wy = y + lx * sin_t + ly * cos_t
        world_pts.append((wx, wy))
    return world_pts


def run_scenario(
    name: str,
    sim: PushTSim,
    policy: any,
    controller: TacticalHierarchicalController,
    init_block: list,
    init_agent: list,
    goal_teleport: dict = None,
    max_steps: int = 350,
):
    controller.reset()
    sim.reset()
    bx, by, bth = init_block
    ax, ay = init_agent
    sim.env.unwrapped._set_state([ax, ay, bx, by, bth])
    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

    cov_eval = PushTCoverageEvaluator(
        goal_x=controller.goal_x,
        goal_y=controller.goal_y,
        goal_theta=controller.goal_theta
    )

    init_cov = cov_eval.compute_coverage(obs["block_pose"][0], obs["block_pose"][1], obs["block_pose"][2])
    agent_traj = [obs["agent_pos_px"].copy()]
    block_traj = [obs["block_pose"][:2].copy()]
    cov_history = [init_cov * 100.0]
    phases = []

    init_dist = math.hypot(bx - controller.goal_x, by - controller.goal_y)
    init_dth = abs((bth - controller.goal_theta + math.pi) % (2.0 * math.pi) - math.pi)
    max_cov = init_cov

    for step in range(max_steps):
        # Handle dynamic goal teleport if configured
        if goal_teleport and step == goal_teleport["step"]:
            gx, gy, gth = goal_teleport["pose"]
            controller.set_goal(gx, gy, gth)
            cov_eval.set_goal(gx, gy, gth)

        act_px, status, info = controller.get_action(obs, sim, policy)
        cur_phase = info.get("phase", "UNKNOWN")
        phases.append(cur_phase)

        # Rate limiting
        curr_agent_px = obs["agent_pos_px"]
        delta = act_px - curr_agent_px
        dist = np.linalg.norm(delta)
        if dist > 35.0:
            act_px = curr_agent_px + (delta / dist) * 35.0

        act_norm = normalize_pos(act_px)
        obs, r, term, trunc, _ = sim.step(act_norm, normalized_action=True)

        bp = obs["block_pose"]
        cov = cov_eval.compute_coverage(bp[0], bp[1], bp[2])
        agent_traj.append(obs["agent_pos_px"].copy())
        block_traj.append(bp[:2].copy())
        cov_history.append(cov * 100.0)
        if cov > max_cov:
            max_cov = cov

        if cur_phase == "LOCKED":
            # Allow a few steps to verify safe retreat
            if step > 20 and all(p == "LOCKED" for p in phases[-6:]):
                break

    final_pose = obs["block_pose"]
    final_dist = math.hypot(final_pose[0] - controller.goal_x, final_pose[1] - controller.goal_y)
    final_dth = abs((final_pose[2] - controller.goal_theta + math.pi) % (2.0 * math.pi) - math.pi)

    return {
        "name": name,
        "init_block": init_block,
        "final_block": [float(final_pose[0]), float(final_pose[1]), float(final_pose[2])],
        "init_dist": round(init_dist, 1),
        "final_dist": round(final_dist, 1),
        "init_dth_deg": round(math.degrees(init_dth), 1),
        "final_dth_deg": round(math.degrees(final_dth), 1),
        "max_cov": round(max_cov * 100.0, 1),
        "final_cov": round(obs["coverage"] * 100.0, 1),
        "steps": len(cov_history) - 1,
        "agent_traj": np.array(agent_traj),
        "block_traj": np.array(block_traj),
        "cov_history": cov_history,
        "phases": phases,
        "final_obs": obs,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"BẮT ĐẦU CHƠI THỬ CÁC TÌNH HUỐNG KIỂM THỬ THỰC TẾ TRÊN DEVICE: {device.upper()}")
    print("=" * 80)

    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    policy = build_policy(config, device)
    controller = TacticalHierarchicalController()
    sim = PushTSim(render_size=(96, 96), max_steps=400)

    scenarios = [
        {
            "name": "1. Chuẩn mực (Góc lệch nhỏ 35°, cự ly vừa)",
            "block": [205.0, 220.0, math.radians(10.0)],
            "agent": [130.0, 310.0],
            "teleport": None,
        },
        {
            "name": "2. Cực đoan (Lật ngược 165° - Khó nhất OOD)",
            "block": [320.0, 230.0, math.radians(-120.0)],
            "agent": [180.0, 380.0],
            "teleport": None,
        },
        {
            "name": "3. Xa bàn cờ (Tọa độ góc xa 200px)",
            "block": [140.0, 140.0, math.radians(70.0)],
            "agent": [380.0, 380.0],
            "teleport": None,
        },
        {
            "name": "4. Sát mép tường (Góc dưới biên 166, 384)",
            "block": [166.0, 384.0, math.radians(136.0)],
            "agent": [120.0, 250.0],
            "teleport": None,
        },
        {
            "name": "5. Mục tiêu di động (Kéo đích thời gian thực)",
            "block": [220.0, 240.0, math.radians(40.0)],
            "agent": [150.0, 330.0],
            "teleport": {"step": 60, "pose": [320.0, 220.0, math.radians(45.0)]},
        },
    ]

    results = []
    for sc in scenarios:
        print(f"\n▶ Đang chơi thử: {sc['name']}...")
        res = run_scenario(
            sc["name"],
            sim,
            policy,
            controller,
            sc["block"],
            sc["agent"],
            goal_teleport=sc["teleport"],
            max_steps=350,
        )
        results.append(res)
        print(f"  + Kết quả: IoU tối đa = {res['max_cov']}% | IoU kết thúc = {res['final_cov']}%")
        print(f"  + Vị trí: Cách đích ban đầu {res['init_dist']}px -> còn {res['final_dist']}px")
        print(f"  + Góc xoay: Lệch ban đầu {res['init_dth_deg']}° -> còn {res['final_dth_deg']}°")
        print(f"  + Số bước hoàn thành: {res['steps']} bước")

    # -------------------------------------------------------------
    # PLOT MULTI-PANEL REPORT FIGURE
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor("#1e1e24")

    # Row 1: 2D Spatial Trajectories
    for i, res in enumerate(results):
        ax = axes[0, i]
        ax.set_facecolor("#2b2b36")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)  # Invert Y to match screen coords
        ax.set_aspect("equal")

        # Goal target T
        goal_x, goal_y, goal_th = 256.0, 256.0, math.pi / 4.0
        if "Mục tiêu di động" in res["name"]:
            goal_x, goal_y, goal_th = 320.0, 220.0, math.radians(45.0)

        goal_pts = get_t_block_vertices(goal_x, goal_y, goal_th)
        goal_patch = patches.Polygon(goal_pts, closed=True, facecolor="#4ade80", alpha=0.35, edgecolor="#22c55e", lw=2, linestyle="--")
        ax.add_patch(goal_patch)

        # Initial T
        init_bx, init_by, init_bth = res["init_block"]
        init_pts = get_t_block_vertices(init_bx, init_by, init_bth)
        init_patch = patches.Polygon(init_pts, closed=True, facecolor="#64748b", alpha=0.4, edgecolor="#94a3b8", lw=1.5)
        ax.add_patch(init_patch)

        # Final T
        fin_bx, fin_by, fin_bth = res["final_block"]
        fin_pts = get_t_block_vertices(fin_bx, fin_by, fin_bth)
        fin_patch = patches.Polygon(fin_pts, closed=True, facecolor="#38bdf8", alpha=0.7, edgecolor="#0284c7", lw=2)
        ax.add_patch(fin_patch)

        # Plot agent path
        atraj = res["agent_traj"]
        ax.plot(atraj[:, 0], atraj[:, 1], color="#f59e0b", lw=2.0, alpha=0.85, label="Agent Path")
        ax.scatter(atraj[0, 0], atraj[0, 1], color="#ef4444", s=60, zorder=5, label="Agent Start")
        ax.scatter(atraj[-1, 0], atraj[-1, 1], color="#10b981", s=70, zorder=5, label="Agent Stop")

        # Block CoG path
        btraj = res["block_traj"]
        ax.plot(btraj[:, 0], btraj[:, 1], color="#38bdf8", lw=2.5, linestyle=":", alpha=0.9, label="Block CoG")

        ax.set_title(f"{res['name'][:18]}...\nPeak IoU: {res['max_cov']}%", color="#f8fafc", fontsize=10, weight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        if i == 0:
            ax.legend(loc="upper left", fontsize=7, facecolor="#1e1e24", edgecolor="#475569", labelcolor="#f8fafc")

    # Row 2: Coverage vs Steps
    for i, res in enumerate(results):
        ax = axes[1, i]
        ax.set_facecolor("#2b2b36")
        steps = np.arange(len(res["cov_history"]))
        covs = res["cov_history"]

        ax.plot(steps, covs, color="#38bdf8", lw=2.5, label="IoU (%)")
        ax.axhline(res["max_cov"], color="#f59e0b", linestyle="--", alpha=0.8, label=f"Peak: {res['max_cov']}%")
        ax.axhline(78.0, color="#10b981", linestyle=":", alpha=0.6, label="Success (78%)")

        ax.set_ylim(-5, 105)
        ax.set_xlim(0, max(len(steps), 100))
        ax.set_xlabel("Simulation Steps", color="#94a3b8", fontsize=9)
        if i == 0:
            ax.set_ylabel("Coverage (IoU %)", color="#94a3b8", fontsize=9)
        ax.set_title(f"Steps: {res['steps']} | Final: {res['final_cov']}%", color="#f8fafc", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, facecolor="#1e1e24", edgecolor="#475569", labelcolor="#f8fafc")

    plt.suptitle("ĐÁNH GIÁ CHƠI THỬ ĐA TRƯỜNG HỢP: KHẢ NĂNG THAO TÁC CỦA WORLD MODEL + DIT", color="#38bdf8", fontsize=14, weight="bold")
    plt.tight_layout()

    out_plot = "results/E2_handcrafted_controllers/play_test_scenarios.png"
    plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n✓ Đã xuất ảnh biểu đồ tổng hợp kết quả chơi thử: {out_plot}")

    # Save JSON summary
    summary_data = []
    for r in results:
        summary_data.append({
            "name": r["name"],
            "init_dist": r["init_dist"],
            "final_dist": r["final_dist"],
            "init_dth_deg": r["init_dth_deg"],
            "final_dth_deg": r["final_dth_deg"],
            "max_cov": r["max_cov"],
            "final_cov": r["final_cov"],
            "steps": r["steps"],
        })
    with open("results/E2_handcrafted_controllers/play_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print("✓ Đã lưu bảng tổng hợp: results/E2_handcrafted_controllers/play_test_summary.json")


if __name__ == "__main__":
    main()

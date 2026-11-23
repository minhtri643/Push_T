"""Interactive Push-T UI: Drag-and-Drop Placement & Autonomous DiT Pushing Demo.

Features:
1. Place Mode:
   - Left Click & Drag: Move the T-block anywhere on the 512x512 workspace.
   - Mouse Scroll or [Q]/[E] keys: Rotate the T-block angle.
   - Right Click: Place the robot pusher anywhere.
   - [SPACE] or Click 'START': Launch autonomous DiT-guided pushing.
2. Autonomous Mode:
   - Real-time closed-loop execution of World Model + DiT Planner + Residual ID.
   - Live trajectory visualization: preview DiT planned path in real-time.
   - Live telemetry HUD: coverage %, distance to goal, and status.
   - Stops with SUCCESS when coverage > 95% (gym-pusht success criterion).
3. Heuristics (OFF by default, same as the standard evaluation):
   - [H] toggles the stance-navigation heuristic, [L] toggles the target lock.
   - With the retrained model both heuristics lower performance (see results/legacy_protocol/E1_augfix).
4. Reset:
   - [R] or Click 'RESET': Returns to placement mode for testing new locations.

Usage:
    python demo.py [--stance] [--early-stop] [--ckpt-dir DIR]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import math
import numpy as np
import pygame
import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim
from controllers.tactical_controller import TacticalHierarchicalController
from controllers.wm_lookahead import (
    WorldModelLookaheadOptimizer,
    compute_safe_retreat,
)
from pusht_wm_dit.policy import (
    SUCCESS_THRESHOLD,
    build_policy,
    postprocess_action,
    resolve_device,
    should_lock,
)
from pusht_wm_dit.utils.normalizer import (
    four_d_to_pose,
    normalize_pos,
    unnormalize_pos,
)


def run_interactive_demo(args: argparse.Namespace):
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = resolve_device(config)

    print("=" * 70)
    print(f"LAUNCHING INTERACTIVE PUSH-T DEMO ON DEVICE: {device.upper()}")
    print("=" * 70)

    # 1. Load Trained Models (World Model -> DiT planner -> Inverse Dynamics)
    print("Loading models...")
    policy = build_policy(config, device, ckpt_dir=args.ckpt_dir)

    # 2. Setup Environment
    max_auto_steps = config["env"]["max_steps"]
    sim = PushTSim(env_name=config["env"]["env_name"], render_size=(96, 96), max_steps=100000)
    obs, info = sim.reset(seed=42)

    # 3. Setup Pygame Window (512x512 Canvas + 320px Side Panel)
    pygame.init()
    pygame.font.init()

    SIM_SIZE = 512
    PANEL_WIDTH = 320
    WINDOW_WIDTH = SIM_SIZE + PANEL_WIDTH
    WINDOW_HEIGHT = SIM_SIZE

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Push-T: Interactive Placement & DiT Autonomous Control")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)
    font_bold = pygame.font.SysFont("Segoe UI", 14, bold=True)
    font_normal = pygame.font.SysFont("Segoe UI", 13)
    font_small = pygame.font.SysFont("Segoe UI", 11)

    # State variables
    mode = "PLACE"  # 'PLACE' or 'AUTO'
    dragging_block = False
    drag_offset = (0, 0)
    current_angle = obs["block_pose"][2]
    max_coverage = obs["coverage"]
    auto_step_count = 0
    peak_step = None
    planned_path_points = []
    tact_info = {}
    status_msg = "Drag green T to set target, grey T to set block start"

    # Target Goal state variables (Real-Time Dynamic Target Chasing)
    goal_x = 256.0
    goal_y = 256.0
    goal_theta = math.pi / 4.0
    dragging_goal = False
    goal_drag_offset = (0.0, 0.0)

    # Button Rectangles
    btn_start_rect = pygame.Rect(SIM_SIZE + 20, 415, 130, 36)
    btn_reset_rect = pygame.Rect(SIM_SIZE + 170, 415, 130, 36)
    btn_random_block_rect = pygame.Rect(SIM_SIZE + 20, 458, 130, 34)
    btn_random_goal_rect = pygame.Rect(SIM_SIZE + 170, 458, 130, 34)

    exec_horizon = config["evaluation"]["exec_horizon"]
    pending_actions = []
    use_stance = False
    use_lock = False
    use_tactical = False
    use_guard = False
    use_lookahead = False
    wm_optimizer = WorldModelLookaheadOptimizer()
    tactical_ctrl = TacticalHierarchicalController(
        enable_collision_guard=False,
        enable_wm_lookahead=False,
        enable_mppi=False,
    )

    running = True
    while running:
        # ---------------------------------------------------------
        # EVENT HANDLING
        # ---------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if mode == "PLACE":
                        mode = "AUTO"
                        max_coverage = obs["coverage"]
                        auto_step_count = 0
                        peak_step = None
                        pending_actions.clear()
                        status_msg = "Autonomous Planning: Chạy liên tục đến khi đạt đỉnh..."
                    else:
                        mode = "PLACE"
                        status_msg = "Paused. Drag to reposition."
                elif event.key == pygame.K_r:
                    mode = "PLACE"
                    obs, info = sim.reset()
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.reset()
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    current_angle = obs["block_pose"][2]
                    max_coverage = obs["coverage"]
                    auto_step_count = 0
                    peak_step = None
                    planned_path_points.clear()
                    status_msg = "Reset to default. Drag to customize."
                elif event.key in (pygame.K_z, pygame.K_c):
                    goal_theta += -0.15 if event.key == pygame.K_z else 0.15
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    status_msg = f"Target Goal Angle: {math.degrees(goal_theta):.0f}°"
                elif event.key == pygame.K_t:
                    use_tactical = not use_tactical
                    tactical_ctrl.reset()
                    status_msg = f"Tactical Strategy (Orient->Align->Push) {'ON' if use_tactical else 'OFF'}"
                elif event.key == pygame.K_m:
                    tactical_ctrl.enable_mppi = not tactical_ctrl.enable_mppi
                    status_msg = f"World Model MPPI Brain {'ON (Physics Imagination)' if tactical_ctrl.enable_mppi else 'OFF'}"
                elif event.key == pygame.K_w:
                    use_lookahead = not use_lookahead
                    tactical_ctrl.enable_wm_lookahead = use_lookahead
                    status_msg = f"WM Lookahead Optimal Stopping {'ON' if use_lookahead else 'OFF'}"
                elif event.key == pygame.K_g:
                    use_guard = not use_guard
                    tactical_ctrl.enable_collision_guard = use_guard
                    status_msg = f"Non-Penetration Guard {'ON' if use_guard else 'OFF'}"
                elif event.key == pygame.K_h:
                    use_stance = not use_stance
                    status_msg = f"Stance heuristic {'ON' if use_stance else 'OFF'}"
                elif event.key == pygame.K_l:
                    use_lock = not use_lock
                    status_msg = f"Target lock {'ON' if use_lock else 'OFF'}"
                elif event.key in (pygame.K_q, pygame.K_e) and mode == "PLACE":
                    current_angle += -0.15 if event.key == pygame.K_q else 0.15
                    # exact=False: rotate in place about the block's center of gravity
                    obs = sim.set_state(obs["agent_pos_px"], (obs["block_pose"][0], obs["block_pose"][1], current_angle), exact=False)

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                dist_to_goal = math.hypot(mx - goal_x, my - goal_y)
                if dist_to_goal < 85 or dragging_goal:
                    goal_theta += event.y * 0.10
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    status_msg = f"Target Goal Angle: {math.degrees(goal_theta):.0f}°"
                elif mode == "PLACE":
                    current_angle += event.y * 0.10
                    obs = sim.set_state(obs["agent_pos_px"], (obs["block_pose"][0], obs["block_pose"][1], current_angle), exact=False)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Button clicks
                if btn_start_rect.collidepoint(mx, my):
                    if mode == "PLACE":
                        mode = "AUTO"
                        max_coverage = obs["coverage"]
                        auto_step_count = 0
                        pending_actions.clear()
                        status_msg = "Autonomous DiT Planning Running..."
                    else:
                        mode = "PLACE"
                        status_msg = "Paused. Drag to reposition."
                elif btn_reset_rect.collidepoint(mx, my):
                    mode = "PLACE"
                    obs, info = sim.reset()
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.reset()
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    current_angle = obs["block_pose"][2]
                    max_coverage = obs["coverage"]
                    auto_step_count = 0
                    peak_step = None
                    planned_path_points.clear()
                    status_msg = "Reset. Drag block to place."
                elif btn_random_block_rect.collidepoint(mx, my):
                    mode = "PLACE"
                    tactical_ctrl.reset()
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    rx = np.random.uniform(100, 400)
                    ry = np.random.uniform(120, 420)
                    ra = np.random.uniform(-math.pi, math.pi)
                    ax = np.random.uniform(80, 430)
                    ay = np.random.uniform(80, 430)
                    current_angle = ra
                    obs = sim.set_state((ax, ay), (rx, ry, ra))
                    max_coverage = obs["coverage"]
                    auto_step_count = 0
                    peak_step = None
                    planned_path_points.clear()
                    status_msg = f"Random Block: ({rx:.0f}, {ry:.0f})"
                elif btn_random_goal_rect.collidepoint(mx, my):
                    goal_x = float(np.random.uniform(120, 380))
                    goal_y = float(np.random.uniform(120, 380))
                    goal_theta = float(np.random.uniform(-math.pi, math.pi))
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    max_coverage = obs["coverage"]
                    peak_step = None
                    status_msg = f"Random Target Goal: ({goal_x:.0f}, {goal_y:.0f})"

                # Canvas interaction (Both PLACE and AUTO mode)
                elif mx < SIM_SIZE:
                    bx, by = obs["block_pose"][0], obs["block_pose"][1]
                    dist_to_goal = math.hypot(mx - goal_x, my - goal_y)
                    dist_to_block = math.hypot(mx - bx, my - by)

                    if event.button == 1:  # Left Click: Grab goal or block
                        if dist_to_goal < 75 or (dist_to_goal < dist_to_block and dist_to_goal < 100) or mode == "AUTO":
                            dragging_goal = True
                            goal_drag_offset = (goal_x - mx, goal_y - my)
                            status_msg = f"Đang kéo mục tiêu chữ T xanh: ({goal_x:.0f}, {goal_y:.0f})"
                        elif mode == "PLACE":
                            if dist_to_block < 80:  # near T-block
                                dragging_block = True
                                drag_offset = (bx - mx, by - my)
                            else:
                                current_angle = obs["block_pose"][2]
                                obs = sim.set_state(obs["agent_pos_px"], (mx, my, current_angle))
                    elif event.button == 3 and mode == "PLACE":  # Right Click: Place agent
                        obs = sim.set_state((mx, my), obs["block_pose"])

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_goal = False
                    dragging_block = False

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if dragging_goal:
                    goal_x = max(50.0, min(SIM_SIZE - 50.0, float(mx + goal_drag_offset[0])))
                    goal_y = max(50.0, min(SIM_SIZE - 50.0, float(my + goal_drag_offset[1])))
                    sim.set_goal_pose([goal_x, goal_y, goal_theta])
                    tactical_ctrl.set_goal(goal_x, goal_y, goal_theta)
                    wm_optimizer.set_goal(goal_x, goal_y, goal_theta)
                    max_coverage = obs["coverage"]
                    if mode == "AUTO":
                        status_msg = f"Mục tiêu di dời -> ({goal_x:.0f}, {goal_y:.0f}) | Cánh tay đang dí theo..."
                elif dragging_block and mode == "PLACE":
                    nx = max(30, min(SIM_SIZE - 30, mx + drag_offset[0]))
                    ny = max(30, min(SIM_SIZE - 30, my + drag_offset[1]))
                    obs = sim.set_state(obs["agent_pos_px"], (nx, ny, current_angle))
                    max_coverage = obs["coverage"]

        # ---------------------------------------------------------
        # AUTONOMOUS INFERENCE STEP
        # ---------------------------------------------------------
        if mode == "AUTO":
            if use_tactical:
                # 3-Phase Tactical Strategy (Orient -> Align -> Push)
                act_world, tact_msg, tact_info = tactical_ctrl.get_action(
                    obs, sim, policy, exec_horizon=exec_horizon
                )
                act_norm = normalize_pos(act_world)
                act_step = postprocess_action(act_norm, obs, use_stance_heuristic=False)

                obs, reward, terminated, truncated, info = sim.step(act_step, normalized_action=True)
                auto_step_count += 1
                cov = obs["coverage"]
                if cov > max_coverage:
                    max_coverage = cov
                status_msg = tact_msg

                # Trajectory preview from tactical controller
                if len(tactical_ctrl.dit_pending_actions) > 0:
                    planned_path_points.clear()
                    for pt in tactical_ctrl.dit_pending_actions:
                        planned_path_points.append((int(pt[0]), int(pt[1])))
                else:
                    planned_path_points.clear()

                if cov > SUCCESS_THRESHOLD or tact_info.get("phase") == "LOCKED":
                    if peak_step is None:
                        peak_step = auto_step_count
                    status_msg = f"🏆 ĐÃ ĐẠT ĐỈNH {max_coverage*100:.1f}% (tại bước {peak_step})! Khóa an toàn [LOCKED]"

            else:
                # Baseline World Model + DiT Planning
                # If no pending actions in the current RHC window, re-plan!
                if len(pending_actions) == 0:
                    # World Model encoding -> DiT plan (step-0 inpainting, best-of-8) -> Inverse Dynamics
                    pred_acts, hat_path_np = policy.plan(obs)

                    # Extract preview points for rendering planned trajectory
                    planned_path_points.clear()
                    for pt in hat_path_np:
                        px_x, px_y, _ = four_d_to_pose(pt)
                        planned_path_points.append((int(px_x), int(px_y)))

                    acts_slice = pred_acts[:exec_horizon]
                    if use_lookahead and hasattr(policy, "wm"):
                        rollout_res = wm_optimizer.evaluate_rollout(
                            policy.wm, obs, acts_slice, policy.device
                        )
                        if rollout_res["decision"] == "LOCK":
                            if peak_step is None:
                                peak_step = auto_step_count
                            mode = "PLACE"
                            status_msg = f"🏆 ĐÃ ĐẠT ĐỈNH {rollout_res['peak_cov']*100:.1f}% (tại bước {peak_step})! [WM Lookahead LOCK]"
                            bx, by = obs["block_pose"][:2]
                            ax, ay = obs["agent_pos_px"]
                            retreat_act = normalize_pos(compute_safe_retreat(np.array([ax, ay]), np.array([bx, by]), retreat_dist=16.0))
                            obs, _, _, _, _ = sim.step(retreat_act, normalized_action=True)
                            pending_actions.clear()
                        elif rollout_res["decision"] == "TRUNCATE":
                            acts_slice = rollout_res["actions_to_execute"]
                            status_msg = f"WM Lookahead: Cắt tỉa vọt lố (k*={rollout_res['optimal_k']}, đỉnh {rollout_res['peak_cov']*100:.1f}%)"

                    for a in acts_slice:
                        pending_actions.append(a)

                # Execute 1 action step per Pygame frame (30 FPS for smooth real-time viewing)
                if len(pending_actions) > 0:
                    # Optional stance heuristic + non-penetration guard + rate limiting
                    act_step = postprocess_action(
                        pending_actions.pop(0),
                        obs,
                        use_stance_heuristic=use_stance,
                        block_body=sim.env.unwrapped.block,
                        use_collision_guard=use_guard,
                    )

                    obs, reward, terminated, truncated, info = sim.step(act_step, normalized_action=True)
                    auto_step_count += 1
                    cov = obs["coverage"]
                    if cov > max_coverage:
                        max_coverage = cov

                    # Stop conditions: ONLY stop when peak is reached (SUCCESS > 95% or target lock)
                    if cov > SUCCESS_THRESHOLD:
                        if peak_step is None:
                            peak_step = auto_step_count
                        mode = "PLACE"
                        status_msg = f"🏆 SUCCESS! Đạt đỉnh ({cov*100:.1f}%) tại bước {peak_step}!"
                        pending_actions.clear()
                    elif use_lock and should_lock(cov, max_coverage):
                        if peak_step is None:
                            peak_step = auto_step_count
                        mode = "PLACE"
                        status_msg = f"🏆 ĐÃ ĐẠT ĐỈNH {max_coverage*100:.1f}% (tại bước {peak_step})! Khóa an toàn."
                        pending_actions.clear()

        # ---------------------------------------------------------
        # RENDERING TO SCREEN
        # ---------------------------------------------------------
        # 1. Render Pymunk simulation canvas (512x512)
        sim_screen = sim.env.unwrapped._draw()
        window.blit(sim_screen, (0, 0))

        # Dynamic Real-Time Target Goal Reticle & Highlight
        ring_col = (0, 255, 140) if dragging_goal else (60, 200, 110)
        gx_int, gy_int = int(goal_x), int(goal_y)
        pygame.draw.circle(window, ring_col, (gx_int, gy_int), 36, 2)
        pygame.draw.circle(window, ring_col, (gx_int, gy_int), 4, 0)
        pygame.draw.line(window, ring_col, (gx_int - 44, gy_int), (gx_int + 44, gy_int), 1)
        pygame.draw.line(window, ring_col, (gx_int, gy_int - 44), (gx_int, gy_int + 44), 1)

        # Dynamic Real-Time Chase Tracking Vector (Yellow line from block CoG to target CoG)
        body = sim.env.unwrapped.block
        cog_world = body.local_to_world((0.0, 45.0))
        goal_cog_x = goal_x - 45.0 * math.sin(goal_theta)
        goal_cog_y = goal_y + 45.0 * math.cos(goal_theta)
        pygame.draw.line(window, (255, 200, 50), (int(cog_world[0]), int(cog_world[1])), (int(goal_cog_x), int(goal_cog_y)), 2)

        # Draw DiT planned trajectory path preview (cyan dashed line & circles)
        if len(planned_path_points) > 1:
            pygame.draw.lines(window, (0, 180, 255), False, planned_path_points, 3)
            for pt in planned_path_points:
                pygame.draw.circle(window, (255, 120, 0), pt, 4)

        # Draw MPPI Imagined Ghost Futures (Glowing multi-trajectory fan)
        if tact_info and "mppi_res" in tact_info:
            mppi_top = tact_info["mppi_res"].get("top_imagined_poses", [])
            for k, traj in enumerate(mppi_top):
                pts = [(int(p[0]), int(p[1])) for p in traj]
                if len(pts) > 1:
                    alpha_col = (0, 230, 255) if k == 0 else (70, 140, 200)
                    pygame.draw.lines(window, alpha_col, False, pts, 2 if k == 0 else 1)
                    pygame.draw.circle(window, (255, 230, 50) if k == 0 else (100, 160, 220), pts[-1], 3)

        # 2. Render Side Control & Telemetry Panel (320px)
        panel_rect = pygame.Rect(SIM_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(window, (24, 28, 36), panel_rect)
        pygame.draw.line(window, (50, 58, 70), (SIM_SIZE, 0), (SIM_SIZE, WINDOW_HEIGHT), 2)

        y_offset = 20
        # Title
        title_surf = font_title.render("Push-T Controller Demo", True, (255, 255, 255))
        window.blit(title_surf, (SIM_SIZE + 20, y_offset))
        y_offset += 25
        sub_text = (
            f"Tactical[T]: {'ON' if use_tactical else 'OFF'} | "
            f"MPPI[M]: {'ON' if tactical_ctrl.enable_mppi else 'OFF'} | "
            f"Lookahead[W]: {'ON' if use_lookahead else 'OFF'}"
        )
        sub_color = (100, 255, 120) if (use_tactical and tactical_ctrl.enable_mppi) else (0, 200, 255)
        sub_surf = font_small.render(sub_text, True, sub_color)
        window.blit(sub_surf, (SIM_SIZE + 20, y_offset))
        y_offset += 30

        # Mode Indicator Pill
        mode_bg = (40, 160, 60) if mode == "AUTO" else (220, 120, 20)
        mode_text = "MODE: AUTONOMOUS PUSHING" if mode == "AUTO" else "MODE: MANUAL PLACEMENT"
        pill_rect = pygame.Rect(SIM_SIZE + 20, y_offset, 280, 26)
        pygame.draw.rect(window, mode_bg, pill_rect, border_radius=4)
        m_surf = font_bold.render(mode_text, True, (255, 255, 255))
        window.blit(m_surf, (SIM_SIZE + 30, y_offset + 5))
        y_offset += 38

        # Telemetry Details
        bx, by, bth = obs["block_pose"][0], obs["block_pose"][1], obs["block_pose"][2]
        ax, ay = obs["agent_pos_px"][0], obs["agent_pos_px"][1]
        dist_goal = math.hypot(bx - goal_x, by - goal_y)

        # Coverage IoU Meter
        if (abs(goal_x - 256.0) > 1.0 or abs(goal_y - 256.0) > 1.0 or abs(goal_theta - math.pi / 4.0) > 0.05):
            curr_cov = tactical_ctrl.wm_optimizer.evaluator.compute_coverage(bx, by, bth) * 100.0
        else:
            curr_cov = obs["coverage"] * 100.0
        cov_lbl = font_bold.render(f"Target Coverage: {curr_cov:.1f}%", True, (255, 255, 255))
        window.blit(cov_lbl, (SIM_SIZE + 20, y_offset))
        y_offset += 22

        # Progress bar
        bar_bg_rect = pygame.Rect(SIM_SIZE + 20, y_offset, 280, 16)
        pygame.draw.rect(window, (45, 52, 65), bar_bg_rect, border_radius=3)
        fill_w = int(280 * min(1.0, curr_cov / 100.0))
        bar_fill_rect = pygame.Rect(SIM_SIZE + 20, y_offset, fill_w, 16)
        bar_color = (40, 210, 80) if curr_cov >= 75 else ((240, 180, 20) if curr_cov >= 40 else (220, 60, 60))
        pygame.draw.rect(window, bar_color, bar_fill_rect, border_radius=3)
        y_offset += 26

        # Telemetry Details
        bx, by, bth = obs["block_pose"][0], obs["block_pose"][1], obs["block_pose"][2]
        ax, ay = obs["agent_pos_px"][0], obs["agent_pos_px"][1]
        dist_goal = math.hypot(bx - goal_x, by - goal_y)

        phase_name = tactical_ctrl.phase if use_tactical else "Standard DiT"
        active_opt = tactical_ctrl.wm_optimizer if use_tactical else wm_optimizer
        step_display = f"{peak_step} (ĐÃ ĐẠT ĐỈNH)" if peak_step is not None else f"{auto_step_count} (Chạy đến khi đạt đỉnh)"
        lines = [
            f"Strategy Phase: {phase_name}",
            f"Peak Coverage:  {max_coverage * 100:.1f}%",
            f"Target Goal:    ({goal_x:.0f}, {goal_y:.0f}) | {math.degrees(goal_theta):.0f} deg",
            f"T-block Pos:    ({bx:.1f}, {by:.1f}) | {math.degrees(bth):.0f} deg",
            f"Goal Distance:  {dist_goal:.1f} px",
            f"Agent Pusher:   ({ax:.1f}, {ay:.1f})",
            f"WM Lookahead:   k*={active_opt.last_best_k} ({active_opt.last_decision})",
            f"Composite Φ:    {active_opt.last_peak_phi:.2f} (Goal: 1.45)",
            f"Steps Executed: {step_display}",
        ]
        for line in lines:
            txt_color = (255, 230, 100) if line.startswith("Strategy Phase") and use_tactical else (180, 190, 205)
            txt_surf = font_normal.render(line, True, txt_color)
            window.blit(txt_surf, (SIM_SIZE + 20, y_offset))
            y_offset += 16
        y_offset += 4

        # Instructions Box
        pygame.draw.rect(window, (32, 38, 48), (SIM_SIZE + 15, y_offset, 290, 120), border_radius=6)
        instr_title = font_bold.render("User Controls:", True, (255, 215, 0))
        window.blit(instr_title, (SIM_SIZE + 25, y_offset + 5))
        y_offset += 20

        instructions = [
            "- Kéo chuột trái T xanh: Di dời mục tiêu (Real-time!)",
            "- Kéo chuột trái T xám: Đặt vị trí khối T (PLACE)",
            "- Chuột phải: Đặt vị trí cánh tay (PLACE)",
            "- Phím Z / C: Xoay góc mục tiêu chữ T xanh",
            "- Cuộn chuột / Q, E: Xoay góc khối T xám",
            "- SPACE: Chạy/Tạm dừng  |  R: Reset",
            "- M: World Model MPPI (Tự suy nghĩ chiến thuật)",
            "- T: Chiến thuật 3-Phase  |  W: WM Lookahead",
            "- G: Non-Penetration Guard (Tránh xuyên vật thể)",
        ]
        for inst in instructions:
            isurf = font_small.render(inst, True, (200, 210, 225))
            window.blit(isurf, (SIM_SIZE + 25, y_offset))
            y_offset += 12

        # Action Buttons
        start_btn_color = (200, 80, 30) if mode == "AUTO" else (30, 140, 60)
        start_btn_txt = "PAUSE [SPACE]" if mode == "AUTO" else "START [SPACE]"
        pygame.draw.rect(window, start_btn_color, btn_start_rect, border_radius=5)
        btn_txt_surf = font_bold.render(start_btn_txt, True, (255, 255, 255))
        window.blit(btn_txt_surf, (btn_start_rect.x + 12, btn_start_rect.y + 9))

        pygame.draw.rect(window, (70, 80, 100), btn_reset_rect, border_radius=5)
        reset_txt_surf = font_bold.render("RESET [R]", True, (255, 255, 255))
        window.blit(reset_txt_surf, (btn_reset_rect.x + 30, btn_reset_rect.y + 9))

        pygame.draw.rect(window, (45, 55, 75), btn_random_block_rect, border_radius=5)
        rnd_b_surf = font_normal.render("RND BLOCK", True, (220, 230, 255))
        window.blit(rnd_b_surf, rnd_b_surf.get_rect(center=btn_random_block_rect.center))

        pygame.draw.rect(window, (35, 75, 55), btn_random_goal_rect, border_radius=5)
        rnd_g_surf = font_normal.render("RND GOAL", True, (150, 255, 180))
        window.blit(rnd_g_surf, rnd_g_surf.get_rect(center=btn_random_goal_rect.center))

        # Bottom Status Message Bar
        pygame.draw.rect(window, (15, 18, 24), (0, SIM_SIZE - 28, SIM_SIZE, 28))
        stat_surf = font_normal.render(f"Status: {status_msg}", True, (255, 230, 120))
        window.blit(stat_surf, (15, SIM_SIZE - 22))

        pygame.display.flip()
        clock.tick(30)

    sim.close()
    pygame.quit()
    print("Interactive Demo closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Push-T demo (WM + DiT + Inverse Dynamics)")
    parser.add_argument("--tactical", action="store_true", help="Start with 3-phase tactical controller ON (toggle: T)")
    parser.add_argument("--no-lookahead", action="store_true", help="Disable World Model lookahead optimal stopping (toggle: W)")
    parser.add_argument("--no-guard", action="store_true", help="Disable rigid-body non-penetration collision guard (toggle: G)")
    parser.add_argument("--stance", action="store_true", help="Start with the stance heuristic ON (toggle: H)")
    parser.add_argument("--early-stop", action="store_true", help="Start with the target lock ON (toggle: L)")
    parser.add_argument("--ckpt-dir", default=None, help="Override checkpoint directory")
    run_interactive_demo(parser.parse_args())

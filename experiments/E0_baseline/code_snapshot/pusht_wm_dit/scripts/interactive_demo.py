"""Interactive Push-T UI: Drag-and-Drop Placement & Autonomous DiT Pushing Demo.

Features:
1. Place Mode:
   - Left Click & Drag: Move the T-block anywhere on the 512x512 workspace.
   - Mouse Scroll or [Q]/[E] keys: Rotate the T-block angle.
   - Right Click: Place the robot pusher anywhere.
   - [SPACE] or Click 'START': Launch autonomous DiT-guided pushing.
2. Autonomous Mode:
   - Real-time closed-loop execution of World Model + DiT Planner + Residual ID on GPU.
   - Live trajectory visualization: preview DiT planned path in real-time.
   - Live telemetry HUD: Coverage IoU %, distance to goal, FDE, and status.
   - Target Lock: automatically locks placement when aligned.
3. Reset:
   - [R] or Click 'RESET': Returns to placement mode for testing new locations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import numpy as np
import pygame
import torch
import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.models.world_model import WorldModel
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.utils.normalizer import (
    normalize_image,
    normalize_pos,
    unnormalize_pos,
    four_d_to_pose,
    pose_to_4d,
)
from pusht_wm_dit.utils.metrics import (
    compute_target_coverage_iou,
    compute_fde_numpy,
)


def run_interactive_demo():
    config_path = "pusht_wm_dit/configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    wm_config = config["world_model"]
    dit_config = config["dit_planner"]
    id_config = config["inverse_dynamics"]

    print("=" * 70)
    print(f"LAUNCHING INTERACTIVE PUSH-T DEMO ON DEVICE: {device.upper()}")
    print("=" * 70)

    # 1. Load Trained Models
    print("Loading models to GPU...")
    wm = WorldModel(
        latent_dim=wm_config["latent_dim"],
        agent_pos_dim=wm_config["agent_pos_dim"],
        pose_dim=wm_config["pose_dim"],
        action_dim=wm_config["action_dim"],
        num_groups=wm_config["num_groups"],
    ).to(device)
    if os.path.exists(wm_config["save_path"]):
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
    if os.path.exists(dit_config["save_path"]):
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
    if os.path.exists(id_config["save_path"]):
        id_ctrl.load_state_dict(torch.load(id_config["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    id_ctrl.eval()

    # 2. Setup Environment
    sim = PushTSim(env_name="gym_pusht/PushT-v0", render_size=(96, 96), max_steps=400)
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
    planned_path_points = []
    status_msg = "Drag T-block to set start position"

    # Button Rectangles
    btn_start_rect = pygame.Rect(SIM_SIZE + 20, 420, 130, 38)
    btn_reset_rect = pygame.Rect(SIM_SIZE + 170, 420, 130, 38)
    btn_random_rect = pygame.Rect(SIM_SIZE + 20, 465, 280, 32)

    exec_horizon = 8
    pending_actions = []

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
                        pending_actions.clear()
                        status_msg = "Autonomous DiT Planning Running..."
                    else:
                        mode = "PLACE"
                        status_msg = "Paused. Drag to reposition."
                elif event.key == pygame.K_r:
                    mode = "PLACE"
                    obs, info = sim.reset()
                    current_angle = obs["block_pose"][2]
                    max_coverage = obs["coverage"]
                    planned_path_points.clear()
                    status_msg = "Reset to default. Drag to customize."
                elif event.key == pygame.K_q and mode == "PLACE":
                    current_angle -= 0.15
                    st = [obs["agent_pos_px"][0], obs["agent_pos_px"][1], obs["block_pose"][0], obs["block_pose"][1], current_angle]
                    sim.env.unwrapped._set_state(st)
                    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())
                elif event.key == pygame.K_e and mode == "PLACE":
                    current_angle += 0.15
                    st = [obs["agent_pos_px"][0], obs["agent_pos_px"][1], obs["block_pose"][0], obs["block_pose"][1], current_angle]
                    sim.env.unwrapped._set_state(st)
                    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

            elif event.type == pygame.MOUSEWHEEL and mode == "PLACE":
                # Scroll wheel rotates block
                current_angle += event.y * 0.10
                st = [obs["agent_pos_px"][0], obs["agent_pos_px"][1], obs["block_pose"][0], obs["block_pose"][1], current_angle]
                sim.env.unwrapped._set_state(st)
                obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

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
                    current_angle = obs["block_pose"][2]
                    max_coverage = obs["coverage"]
                    planned_path_points.clear()
                    status_msg = "Reset. Drag block to place."
                elif btn_random_rect.collidepoint(mx, my):
                    mode = "PLACE"
                    rx = np.random.uniform(100, 400)
                    ry = np.random.uniform(120, 420)
                    ra = np.random.uniform(-math.pi, math.pi)
                    ax = np.random.uniform(80, 430)
                    ay = np.random.uniform(80, 430)
                    current_angle = ra
                    st = [ax, ay, rx, ry, ra]
                    sim.env.unwrapped._set_state(st)
                    obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())
                    max_coverage = obs["coverage"]
                    planned_path_points.clear()
                    status_msg = f"Random position set: ({rx:.0f}, {ry:.0f})"

                # Canvas interaction (only in PLACE mode)
                elif mode == "PLACE" and mx < SIM_SIZE:
                    if event.button == 1:  # Left Click: Grab block
                        bx, by = obs["block_pose"][0], obs["block_pose"][1]
                        dist = math.hypot(mx - bx, my - by)
                        if dist < 80:  # near T-block
                            dragging_block = True
                            drag_offset = (bx - mx, by - my)
                        else:
                            # Direct place block at click
                            current_angle = obs["block_pose"][2]
                            st = [obs["agent_pos_px"][0], obs["agent_pos_px"][1], mx, my, current_angle]
                            sim.env.unwrapped._set_state(st)
                            obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())
                    elif event.button == 3:  # Right Click: Place agent
                        st = [mx, my, obs["block_pose"][0], obs["block_pose"][1], obs["block_pose"][2]]
                        sim.env.unwrapped._set_state(st)
                        obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_block = False

            elif event.type == pygame.MOUSEMOTION and dragging_block and mode == "PLACE":
                mx, my = event.pos
                nx = max(30, min(SIM_SIZE - 30, mx + drag_offset[0]))
                ny = max(30, min(SIM_SIZE - 30, my + drag_offset[1]))
                st = [obs["agent_pos_px"][0], obs["agent_pos_px"][1], nx, ny, current_angle]
                sim.env.unwrapped._set_state(st)
                obs = sim._format_observation(sim.env.unwrapped.get_obs(), sim.env.unwrapped._get_info())
                max_coverage = obs["coverage"]

        # ---------------------------------------------------------
        # AUTONOMOUS INFERENCE STEP
        # ---------------------------------------------------------
        if mode == "AUTO":
            # If no pending actions in the current RHC window, re-plan!
            if len(pending_actions) == 0:
                frame_tensor = torch.from_numpy(normalize_image(obs["pixels"])).float().unsqueeze(0).to(device)
                agent_tensor = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
                curr_pose_t = torch.from_numpy(obs["block_pose_4d"]).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    # 1. World Model Latent Encoding
                    s_t = wm.encode(frame_tensor, agent_tensor)

                    # 2. DiT Planner (with Step-0 Inpainting & Best-of-8)
                    hat_path = dit.sample(
                        s_t,
                        num_inference_steps=10,
                        current_pose_4d=curr_pose_t,
                        best_of_n=8,
                    )

                    # 3. Residual Inverse Dynamics Controller
                    actions = id_ctrl(s_t, hat_path, agent_pos=agent_tensor)

                pred_acts = actions[0].cpu().numpy()
                hat_path_np = hat_path[0].cpu().numpy()

                # Extract preview points for rendering planned trajectory
                planned_path_points.clear()
                for pt in hat_path_np:
                    px_x, px_y, _ = four_d_to_pose(pt)
                    planned_path_points.append((int(px_x), int(px_y)))

                for a_idx in range(min(exec_horizon, len(pred_acts))):
                    pending_actions.append(pred_acts[a_idx])

            # Execute 1 action step per Pygame frame (30 FPS for smooth real-time viewing)
            if len(pending_actions) > 0:
                act_step = pending_actions.pop(0)

                # Smart Stance Navigation: ensure pusher doesn't ram the block away when starting in front
                bx, by = obs["block_pose"][0], obs["block_pose"][1]
                ax, ay = obs["agent_pos_px"][0], obs["agent_pos_px"][1]
                gx, gy = 256.0, 256.0
                dist_g = math.hypot(gx - bx, gy - by)

                if dist_g > 35.0:
                    ux = (gx - bx) / (dist_g + 1e-6)
                    uy = (gy - by) / (dist_g + 1e-6)
                    rx = ax - bx
                    ry = ay - by
                    alignment = rx * ux + ry * uy
                    if alignment > 20.0:  # Agent is between block and goal!
                        perp_x = -uy
                        perp_y = ux
                        side = 1.0 if (rx * perp_x + ry * perp_y) >= 0 else -1.0
                        wp_x = bx + side * perp_x * 75.0 - ux * 25.0
                        wp_y = by + side * perp_y * 75.0 - uy * 25.0
                        act_step = normalize_pos(np.array([wp_x, wp_y], dtype=np.float32))

                # Rate limiting
                act_px = unnormalize_pos(act_step)
                curr_agent_px = obs["agent_pos_px"]
                delta = act_px - curr_agent_px
                dist = np.linalg.norm(delta)
                if dist > 40.0:
                    act_px = curr_agent_px + (delta / dist) * 40.0
                    act_step = normalize_pos(act_px)

                obs, reward, terminated, truncated, info = sim.step(act_step, normalized_action=True)
                auto_step_count += 1
                cov = obs["coverage"]
                if cov > max_coverage:
                    max_coverage = cov

                # Target lock & settling
                if cov >= 0.85:
                    mode = "PLACE"
                    status_msg = f"SUCCESS! Goal Reached ({cov*100:.1f}%)"
                    pending_actions.clear()
                elif max_coverage >= 0.75 and cov < max_coverage - 0.04:
                    mode = "PLACE"
                    status_msg = f"TARGET LOCKED! Peak Placement ({max_coverage*100:.1f}%)"
                    pending_actions.clear()
                elif auto_step_count >= 300:
                    mode = "PLACE"
                    status_msg = "Max steps reached. Place again."
                    pending_actions.clear()

        # ---------------------------------------------------------
        # RENDERING TO SCREEN
        # ---------------------------------------------------------
        # 1. Render Pymunk simulation canvas (512x512)
        sim_screen = sim.env.unwrapped._draw()
        window.blit(sim_screen, (0, 0))

        # Draw DiT planned trajectory path preview (cyan dashed line & circles)
        if len(planned_path_points) > 1:
            pygame.draw.lines(window, (0, 180, 255), False, planned_path_points, 3)
            for pt in planned_path_points:
                pygame.draw.circle(window, (255, 120, 0), pt, 4)

        # 2. Render Side Control & Telemetry Panel (320px)
        panel_rect = pygame.Rect(SIM_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(window, (24, 28, 36), panel_rect)
        pygame.draw.line(window, (50, 58, 70), (SIM_SIZE, 0), (SIM_SIZE, WINDOW_HEIGHT), 2)

        y_offset = 20
        # Title
        title_surf = font_title.render("Push-T Controller Demo", True, (255, 255, 255))
        window.blit(title_surf, (SIM_SIZE + 20, y_offset))
        y_offset += 25
        sub_surf = font_small.render("World Model + 1D DiT Planner (CUDA)", True, (0, 200, 255))
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

        # Coverage IoU Meter
        curr_cov = obs["coverage"] * 100
        cov_lbl = font_bold.render(f"Target Coverage IoU: {curr_cov:.1f}%", True, (255, 255, 255))
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
        dist_goal = math.hypot(bx - 256.0, by - 256.0)

        lines = [
            f"Peak Coverage:  {max_coverage * 100:.1f}%",
            f"T-block Pos:    ({bx:.1f}, {by:.1f})",
            f"T-block Angle:  {bth:.2f} rad ({math.degrees(bth):.0f} deg)",
            f"Goal Distance:  {dist_goal:.1f} px",
            f"Agent Pusher:   ({ax:.1f}, {ay:.1f})",
            f"Steps Executed: {auto_step_count}",
        ]
        for line in lines:
            txt_surf = font_normal.render(line, True, (180, 190, 205))
            window.blit(txt_surf, (SIM_SIZE + 20, y_offset))
            y_offset += 20
        y_offset += 10

        # Instructions Box
        pygame.draw.rect(window, (32, 38, 48), (SIM_SIZE + 15, y_offset, 290, 120), border_radius=6)
        instr_title = font_bold.render("User Controls:", True, (255, 215, 0))
        window.blit(instr_title, (SIM_SIZE + 25, y_offset + 8))
        y_offset += 28

        instructions = [
            "- Left Click & Drag: Move T-block",
            "- Scroll / Q, E keys: Rotate Angle",
            "- Right Click: Place Pusher (Agent)",
            "- SPACE / START: Run DiT Policy",
            "- R / RESET: Pause & Reposition",
        ]
        for inst in instructions:
            isurf = font_small.render(inst, True, (200, 210, 225))
            window.blit(isurf, (SIM_SIZE + 25, y_offset))
            y_offset += 17

        # Action Buttons
        start_btn_color = (200, 80, 30) if mode == "AUTO" else (30, 140, 60)
        start_btn_txt = "PAUSE [SPACE]" if mode == "AUTO" else "START [SPACE]"
        pygame.draw.rect(window, start_btn_color, btn_start_rect, border_radius=5)
        btn_txt_surf = font_bold.render(start_btn_txt, True, (255, 255, 255))
        window.blit(btn_txt_surf, (btn_start_rect.x + 12, btn_start_rect.y + 10))

        pygame.draw.rect(window, (70, 80, 100), btn_reset_rect, border_radius=5)
        reset_txt_surf = font_bold.render("RESET [R]", True, (255, 255, 255))
        window.blit(reset_txt_surf, (btn_reset_rect.x + 30, btn_reset_rect.y + 10))

        pygame.draw.rect(window, (45, 55, 75), btn_random_rect, border_radius=5)
        rnd_txt_surf = font_bold.render("🎲 SET RANDOM POSITION", True, (220, 230, 255))
        window.blit(rnd_txt_surf, (btn_random_rect.x + 45, btn_random_rect.y + 7))

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
    run_interactive_demo()

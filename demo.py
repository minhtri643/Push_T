"""Push-T interactive demo: place the T-block, press Space, watch World Model + DiT push it.

Deliberately simple: one board, one number (coverage), two options.

    Chuột trái kéo khối T · cuộn chuột (hoặc Q/E) để xoay · chuột phải đặt đầu đẩy
    Space chạy/tạm dừng · R đặt lại · N vị trí ngẫu nhiên
    M chọn hành động bằng World Model · V chỉ dùng ảnh (tự tắt M) · G hiện dự đoán của World Model · Esc thoát

The faint T on the board is where the World Model imagines the block will be after the actions
that are about to run; the panel shows how far that imagination was from what really happened.

    python demo.py [--ckpt-dir DIR] [--seed N]
The previous demo with the hand-written controllers is in experiments/E2_handcrafted_controllers/code/legacy_interactive_demo.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import math

import numpy as np
import pygame
import yaml

from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.policy import SUCCESS_THRESHOLD, build_policy, load_vision_bundle, postprocess_action, resolve_device
from pusht_wm_dit.utils.normalizer import four_d_to_pose

BOARD, PANEL = 512, 288
W, H = BOARD + PANEL, BOARD
PAD = 24
TEE = [(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)]
C = {
    "bg": (246, 247, 249), "line": (226, 229, 234), "text": (31, 41, 51), "muted": (107, 119, 133),
    "accent": (52, 87, 213), "accent_soft": (224, 231, 251), "ok": (35, 128, 74), "ok_soft": (221, 242, 228),
    "warn": (176, 112, 0), "warn_soft": (251, 240, 214), "white": (255, 255, 255), "track": (230, 233, 238),
}
STATUS = {  # mode -> (label, foreground, background)
    "ready": ("Sẵn sàng", C["muted"], C["track"]),
    "running": ("Đang chạy", C["accent"], C["accent_soft"]),
    "paused": ("Tạm dừng", C["warn"], C["warn_soft"]),
    "success": ("Thành công", C["ok"], C["ok_soft"]),
    "timeout": ("Hết lượt", C["warn"], C["warn_soft"]),
}


def vn(x: float, d: int = 1) -> str:
    """Vietnamese decimal comma."""
    return f"{x:.{d}f}".replace(".", ",")


def tee_points(x, y, th):
    x, y, th = float(x), float(y), float(th)
    c, s = math.cos(th), math.sin(th)
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in TEE]


class UI:
    def __init__(self, window):
        self.win = window
        f = lambda size, bold=False: pygame.font.SysFont("Segoe UI", size, bold=bold)  # noqa: E731
        self.f_title, self.f_big, self.f_body = f(22, True), f(52, True), f(14)
        self.f_small, self.f_label, self.f_btn = f(12), f(11, True), f(14, True)
        self.buttons = {}
        self.toggles = {}

    def text(self, s, font, color, pos, anchor="topleft"):
        surf = font.render(s, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.win.blit(surf, rect)
        return rect

    def pill(self, rect, color, radius=None):
        pygame.draw.rect(self.win, color, rect, border_radius=radius if radius is not None else rect.height // 2)

    def button(self, key, rect, label, primary=False, hover=False):
        bg = C["accent"] if primary else C["white"]
        if hover:
            bg = tuple(max(0, v - 14) for v in bg)
        pygame.draw.rect(self.win, bg, rect, border_radius=8)
        if not primary:
            pygame.draw.rect(self.win, C["line"], rect, 1, border_radius=8)
        self.text(label, self.f_btn, C["white"] if primary else C["text"], rect.center, "center")
        self.buttons[key] = rect

    def toggle(self, key, y, label, hint, on, enabled=True):
        x0, x1 = BOARD + PAD, W - PAD
        r = self.text(label, self.f_body, C["text"] if enabled else C["muted"], (x0, y))
        self.text(hint, self.f_small, C["muted"], (r.right + 6, y + 3))
        sw = pygame.Rect(x1 - 40, y, 40, 22)
        self.pill(sw, C["accent"] if (on and enabled) else C["track"])
        knob_x = sw.right - 11 if on else sw.left + 11
        pygame.draw.circle(self.win, C["white"], (knob_x, sw.centery), 8)
        self.toggles[key] = pygame.Rect(x0, y - 4, x1 - x0, 30)


def run(args):
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg)
    print("Đang nạp mô hình...")
    policy = build_policy(cfg, device, ckpt_dir=args.ckpt_dir, kind="dit_id")
    policy.vision = load_vision_bundle(cfg, device, args.ckpt_dir)
    has_vision = policy.vision is not None
    has_wm = policy.dynamics is not None or has_vision
    ui_state = {"wm": cfg.get("demo", {}).get("action_mode", "id") == "wm" and has_wm,
                "vision": bool(cfg.get("demo", {}).get("vision_only", False)) and has_vision}

    def apply_modes():
        """Vision only: block pose read from the image, actions imagined by the vision World Model."""
        policy.pose_source = "vision" if ui_state["vision"] else "true"
        if ui_state["wm"]:
            policy.action_mode = "vwm" if ui_state["vision"] else ("wm" if policy.dynamics is not None else "id")
        else:
            policy.action_mode = "id"

    apply_modes()
    max_steps = int(cfg.get("demo", {}).get("max_steps", 300))
    exec_h = cfg["evaluation"]["exec_horizon"]

    sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=(96, 96), max_steps=10 ** 9)
    obs, _ = sim.reset(seed=args.seed)

    pygame.init()
    window = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Push-T · World Model + DiT")
    clock = pygame.time.Clock()
    ui = UI(window)
    overlay = pygame.Surface((BOARD, BOARD), pygame.SRCALPHA)

    st = {"mode": "ready", "step": 0, "peak": obs["coverage"], "pending": [], "path": [], "ghost": None,
          "ghost_at": None, "wm_err": None, "show_ghost": has_wm, "drag": None, "angle": float(obs["block_pose"][2]),
          "msg": "Kéo khối T rồi nhấn Space"}

    def new_trial(o, msg):
        st.update(mode="ready", step=0, peak=o["coverage"], pending=[], path=[], ghost=None, ghost_at=None,
                  wm_err=None, angle=float(o["block_pose"][2]), msg=msg)
        return o

    def toggle_vision():
        """Turning "vision only" on also turns World-Model action selection off: with the vision World Model
        the selection lowers success (E5b_jepa__dit_vwm 20% vs E5b_jepa__dit_id_vision 45%, results/benchmark).
        M can still turn it back on; turning vision off restores the previous M setting."""
        ui_state["vision"] = not ui_state["vision"]
        if ui_state["vision"]:
            ui_state["wm_before_vision"], ui_state["wm"] = ui_state["wm"], False
        else:
            ui_state["wm"] = ui_state.pop("wm_before_vision", ui_state["wm"])
        apply_modes()
        st["pending"] = []

    def toggle_run():
        if st["mode"] == "ready":
            policy.reset(obs)
        if st["mode"] in ("ready", "paused"):
            st["mode"], st["msg"] = "running", ""
        elif st["mode"] == "running":
            st["mode"] = "paused"
        else:  # finished: start the same placement again
            st.update(mode="running", step=0, peak=obs["coverage"], pending=[], msg="")

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    toggle_run()
                elif ev.key == pygame.K_r:
                    obs = new_trial(sim.reset()[0], "Đã đặt lại")
                elif ev.key == pygame.K_n:
                    obs = new_trial(sim.set_state(np.random.uniform(90, 420, 2), (*np.random.uniform(110, 400, 2), np.random.uniform(-math.pi, math.pi))), "Vị trí ngẫu nhiên")
                elif ev.key == pygame.K_m and has_wm:
                    ui_state["wm"] = not ui_state["wm"]
                    apply_modes()
                    st["pending"] = []
                elif ev.key == pygame.K_v and has_vision:
                    toggle_vision()
                elif ev.key == pygame.K_g and has_wm:
                    st["show_ghost"] = not st["show_ghost"]
                elif ev.key in (pygame.K_q, pygame.K_e) and st["mode"] != "running":
                    st["angle"] += -0.15 if ev.key == pygame.K_q else 0.15
                    obs = new_trial(sim.set_state(obs["agent_pos_px"], (*obs["block_pose"][:2], st["angle"]), exact=False), "")
            elif ev.type == pygame.MOUSEWHEEL and st["mode"] != "running" and mouse[0] < BOARD:
                st["angle"] += ev.y * 0.10
                obs = new_trial(sim.set_state(obs["agent_pos_px"], (*obs["block_pose"][:2], st["angle"]), exact=False), "")
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
                hit = next((k for k, r in ui.buttons.items() if r.collidepoint(ev.pos)), None) if ev.button == 1 else None
                tog = next((k for k, r in ui.toggles.items() if r.collidepoint(ev.pos)), None) if ev.button == 1 else None
                if hit == "run":
                    toggle_run()
                elif hit == "reset":
                    obs = new_trial(sim.reset()[0], "Đã đặt lại")
                elif hit == "random":
                    obs = new_trial(sim.set_state(np.random.uniform(90, 420, 2), (*np.random.uniform(110, 400, 2), np.random.uniform(-math.pi, math.pi))), "Vị trí ngẫu nhiên")
                elif tog == "wm_actions" and has_wm:
                    ui_state["wm"] = not ui_state["wm"]
                    apply_modes()
                    st["pending"] = []
                elif tog == "vision" and has_vision:
                    toggle_vision()
                elif tog == "ghost" and has_wm:
                    st["show_ghost"] = not st["show_ghost"]
                elif ev.pos[0] < BOARD and st["mode"] != "running":
                    if ev.button == 3:
                        obs = new_trial(sim.set_state(ev.pos, obs["block_pose"]), "")
                    else:
                        bx, by = obs["block_pose"][:2]
                        if math.hypot(ev.pos[0] - bx, ev.pos[1] - by) < 80:
                            st["drag"] = (bx - ev.pos[0], by - ev.pos[1])
                        else:
                            obs = new_trial(sim.set_state(obs["agent_pos_px"], (ev.pos[0], ev.pos[1], st["angle"])), "")
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                st["drag"] = None
            elif ev.type == pygame.MOUSEMOTION and st["drag"] is not None:
                nx = min(max(ev.pos[0] + st["drag"][0], 30), BOARD - 30)
                ny = min(max(ev.pos[1] + st["drag"][1], 30), BOARD - 30)
                obs = new_trial(sim.set_state(obs["agent_pos_px"], (nx, ny, st["angle"])), "")

        # ---- autonomous control: plan every exec_h steps, execute one step per frame ----
        if st["mode"] == "running":
            if not st["pending"]:
                g_all, n_exec = st["ghost"], st["ghost_at"]
                if g_all is not None and n_exec:  # how good was the last imagination?
                    per = max(1, st["ghost_span"] // len(g_all))  # steps covered by one imagined pose
                    j = min(len(g_all), n_exec // per) - 1
                    if j >= 0 and (j + 1) * per == n_exec:
                        g = g_all[j]
                        d = (obs["block_pose"][2] - g[2] + math.pi) % (2 * math.pi) - math.pi
                        st["wm_err"] = (math.hypot(obs["block_pose"][0] - g[0], obs["block_pose"][1] - g[1]), abs(math.degrees(d)))
                actions, path = policy.plan(obs)
                st["pending"] = list(actions[:exec_h])
                st["path"] = [tuple(int(v) for v in four_d_to_pose(p)[:2]) for p in path] if path is not None else []
                st["ghost"] = policy.imagine(obs, np.array(st["pending"])) if has_wm else None
                st["ghost_span"] = len(st["pending"])
                st["ghost_at"] = 0
            obs, _, terminated, _, _ = sim.step(postprocess_action(st["pending"].pop(0), obs), normalized_action=True)
            policy.observe(obs)
            st["step"] += 1
            st["ghost_at"] = (st["ghost_at"] or 0) + 1
            st["peak"] = max(st["peak"], obs["coverage"])
            if terminated or obs["coverage"] > SUCCESS_THRESHOLD:
                st["mode"], st["msg"] = "success", f"Đạt {vn(obs['coverage'] * 100)}% sau {st['step']} bước ({vn(st['step'] / 10)} s)"
                st["pending"] = []
            elif st["step"] >= max_steps:
                st["mode"], st["msg"] = "timeout", f"Dừng ở {max_steps} bước, cao nhất {st['peak'] * 100:.0f}%"
                st["pending"] = []

        # ---- board ----
        window.blit(sim.env.unwrapped._draw(), (0, 0))
        overlay.fill((0, 0, 0, 0))
        if st["show_ghost"] and st["ghost"] is not None:
            g = st["ghost"][-1]
            pts = tee_points(g[0], g[1], g[2])
            pygame.draw.polygon(overlay, (*C["accent"], 38), pts)
            pygame.draw.polygon(overlay, (*C["accent"], 170), pts, 2)
        if len(st["path"]) > 1:
            pygame.draw.lines(overlay, (*C["accent"], 150), False, st["path"], 2)
            for p in st["path"][1:]:
                pygame.draw.circle(overlay, (*C["accent"], 170), p, 2)
        window.blit(overlay, (0, 0))
        if st["msg"]:
            ui.text(st["msg"], ui.f_small, C["muted"], (12, BOARD - 10), "bottomleft")

        # ---- panel ----
        pygame.draw.rect(window, C["bg"], (BOARD, 0, PANEL, H))
        pygame.draw.line(window, C["line"], (BOARD, 0), (BOARD, H))
        x0, x1 = BOARD + PAD, W - PAD
        ui.text("Push-T", ui.f_title, C["text"], (x0, 20))
        ui.text("World Model + DiT", ui.f_small, C["muted"], (x0, 50))
        label, fg, bg = STATUS[st["mode"]]
        chip_w = ui.f_label.size(label)[0] + 22
        chip = pygame.Rect(x1 - chip_w, 24, chip_w, 24)
        ui.pill(chip, bg)
        ui.text(label, ui.f_label, fg, chip.center, "center")

        cov = obs["coverage"]
        ui.text(f"{cov * 100:.0f}%", ui.f_big, C["ok"] if cov > SUCCESS_THRESHOLD else C["text"], (x0 - 2, 76))
        ui.text("độ phủ ô đích", ui.f_small, C["muted"], (x0, 146))
        bar = pygame.Rect(x0, 170, x1 - x0, 8)
        ui.pill(bar, C["track"])
        if cov > 0.005:
            ui.pill(pygame.Rect(x0, 170, max(8, int(bar.width * min(cov, 1.0))), 8), C["ok"] if cov > SUCCESS_THRESHOLD else C["accent"])
        mark = x0 + int(bar.width * SUCCESS_THRESHOLD)
        pygame.draw.line(window, C["muted"], (mark, 166), (mark, 181), 1)
        ui.text("95%", ui.f_small, C["muted"], (mark, 184), "midtop")

        rows = [("Bước", f"{st['step']}  ·  {vn(st['step'] / 10)} s"), ("Cao nhất", f"{st['peak'] * 100:.0f}%")]
        if has_wm:
            rows.append(("WM dự đoán lệch", "—" if st["wm_err"] is None else f"{vn(st['wm_err'][0])} px · {vn(st['wm_err'][1])}°"))
        y = 212
        for k, v in rows:
            ui.text(k, ui.f_body, C["muted"], (x0, y))
            ui.text(v, ui.f_body, C["text"], (x1, y), "topright")
            y += 24
        pygame.draw.line(window, C["line"], (x0, y + 6), (x1, y + 6))

        ui.toggle("wm_actions", y + 18, "Chọn hành động bằng WM", "M", ui_state["wm"], has_wm)
        ui.toggle("vision", y + 52, "Chỉ dùng ảnh", "V", ui_state["vision"], has_vision)
        ui.toggle("ghost", y + 86, "Hiện dự đoán của WM", "G", st["show_ghost"], has_wm)

        by, bx = H - 92, x0
        labels = {"running": "Tạm dừng"}
        for key, text, bw in (("run", labels.get(st["mode"], "Chạy"), 76), ("reset", "Đặt lại", 68), ("random", "Ngẫu nhiên", 88)):
            r = pygame.Rect(bx, by, bw, 36)
            ui.button(key, r, text, primary=(key == "run"), hover=r.collidepoint(mouse))
            bx += bw + 4
        ui.text("Kéo khối · cuộn để xoay · chuột phải: đầu đẩy", ui.f_small, C["muted"], (x0, H - 44))
        ui.text("Space chạy · R đặt lại · N ngẫu nhiên · Esc thoát", ui.f_small, C["muted"], (x0, H - 26))

        pygame.display.flip()
        clock.tick(30)

    sim.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push-T interactive demo (World Model + DiT)")
    parser.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default: config paths)")
    parser.add_argument("--seed", type=int, default=42, help="initial placement")
    run(parser.parse_args())

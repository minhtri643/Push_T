"""Recover the T-block pose (x, y, theta) from a 96x96 Push-T frame by render-and-compare.

lerobot/pusht has frames, pusher positions and actions, but no block pose. The pose is needed as a label
(DiT path, FDE, dynamics model), so it is estimated from the image itself:

  1. a differentiable renderer reproduces how gym_pusht draws a frame (pymunk_override.py):
     white background, LightGreen goal T at (256, 256, pi/4), LightGray walls, RoyalBlue pusher
     (radius 15, lighter core radius 11), and the block as two polygons, each filled with a lighter
     LightSlateGray and outlined 2 px each side with LightSlateGray; the 512x512 canvas is then
     resized to 96x96, i.e. pixel (row i, col j) samples the world point ((j+0.5)*512/96, (i+0.5)*512/96).
     Edges are softened with a sigmoid of width `s` (world px) to model resizing + video compression.
  2. the pusher position is known (observation.state), only the block pose is unknown.
  3. (x, y, theta) is optimised with Adam from 8 starting angles around the centroid of the block-coloured
     pixels, with a coarse-to-fine blur schedule; the start with the lowest robust photometric loss wins.
  4. the 7 colours and the blur width are calibrated once per dataset (`calibrate`), jointly with the poses
     of a few hundred frames, because the AV1 decode shifts colours (e.g. background 248 instead of 255).
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

IMG = 96
WORLD = 512.0
PIX = WORLD / IMG
GOAL_POSE = (256.0, 256.0, math.pi / 4)
OUTLINE = 2.0                                   # outline half-width of the block polygons (world px)
AGENT_R, AGENT_R_IN = 15.0, 11.0
T_CENTROID_Y = (3600 * 15 + 2700 * 75) / 6300   # area centroid of the T in body coordinates: (0, 40.7)
# body-frame outline of the T (same as gym_pusht add_tee with scale 30): bar 120x30 on top of a 30x90 stem
T_VERTS = np.array([[-60, 0], [60, 0], [60, 30], [15, 30], [15, 120], [-15, 120], [-15, 30], [-60, 30]], np.float64)

PALETTE_NAMES = ("background", "goal", "wall", "agent", "agent_inner", "block_edge", "block_fill")
PALETTE0 = np.array([[255, 255, 255], [144, 238, 144], [211, 211, 211], [65, 105, 225], [78, 126, 255],
                     [119, 136, 153], [142.8, 163.2, 183.6]], np.float32) / 255.0
EPS = 0.02                                      # Charbonnier epsilon (colour units in [0, 1])
BORDER = 2                                      # ignore the outermost pixels (walls, codec edge effects)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TeeRenderer:
    """All methods take explicit pixel grids px, py (world px of each sampled pixel) and weights w,
    shaped (B, P) - either the full 96x96 image or a crop around the block."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        c = (torch.arange(IMG, device=device, dtype=torch.float32) + 0.5) * PIX
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.px_img, self.py_img = gx, gy                                             # (96, 96)
        w = torch.zeros(IMG, IMG, device=device)
        w[BORDER:IMG - BORDER, BORDER:IMG - BORDER] = 1
        self.w_img = w

    def full_grid(self, b: int):
        e = lambda t: t.reshape(1, -1).expand(b, -1)
        return e(self.px_img), e(self.py_img), e(self.w_img)

    def crop(self, obs_img: torch.Tensor, centre: torch.Tensor, size: int):
        """obs_img (B, 96, 96, 3), centre (B, 2) world px -> obs (B, P, 3), px, py, w (B, P)."""
        b = len(obs_img)
        top_left = (centre / PIX - size / 2).round().long().clamp(0, IMG - size)    # (B, 2) as (col, row)
        ar = torch.arange(size, device=self.device)
        rows = (top_left[:, 1:2] + ar)[:, :, None].expand(b, size, size)
        cols = (top_left[:, 0:1] + ar)[:, None, :].expand(b, size, size)
        bi = torch.arange(b, device=self.device)[:, None, None].expand(b, size, size)
        f = lambda t: t[rows, cols].reshape(b, -1)
        return obs_img[bi, rows, cols].reshape(b, -1, 3), f(self.px_img), f(self.py_img), f(self.w_img)

    @staticmethod
    def _box_sd(lx, ly, cy, hx, hy):
        qx = lx.abs() - hx
        qy = (ly - cy).abs() - hy
        return torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-8) + torch.maximum(qx, qy).clamp(max=0)

    def tee_sd(self, pose: torch.Tensor, px: torch.Tensor, py: torch.Tensor):
        """Signed distances to the bar and to the stem. pose (..., 3) broadcast against px/py."""
        x, y, th = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
        c, s = torch.cos(th), torch.sin(th)
        dx, dy = px - x, py - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return self._box_sd(lx, ly, 15.0, 60.0, 15.0), self._box_sd(lx, ly, 75.0, 15.0, 45.0)

    def base(self, agent: torch.Tensor, pal: torch.Tensor, s, px, py) -> torch.Tensor:
        """Everything except the block: agent (B, 2), px/py (B, P) -> (B, P, 3)."""
        gA, gB = self.tee_sd(torch.tensor(GOAL_POSE, device=self.device), px, py)
        goal = torch.sigmoid(-torch.minimum(gA, gB) / s)[..., None]
        dwall = torch.stack([(px - 5).abs(), (py - 5).abs(), (px - 506).abs(), (py - 506).abs()]).min(0).values
        wall = torch.sigmoid((2.0 - dwall) / s)[..., None]
        img = pal[0] + goal * (pal[1] - pal[0])
        img = img + wall * (pal[2] - img)
        a = torch.round(agent)
        d = torch.sqrt((px - a[:, 0:1]) ** 2 + (py - a[:, 1:2]) ** 2)                # (B, P)
        img = img + torch.sigmoid((AGENT_R - d) / s)[..., None] * (pal[3] - img)
        img = img + torch.sigmoid((AGENT_R_IN - d) / s)[..., None] * (pal[4] - img)
        return img

    def with_tee(self, base: torch.Tensor, pose: torch.Tensor, pal: torch.Tensor, s, px, py) -> torch.Tensor:
        """base (B, P, 3), pose (B, K, 3) -> (B, K, P, 3). Bar drawn first, then stem (each fill + outline)."""
        img = base[:, None]
        for sd in self.tee_sd(pose, px[:, None], py[:, None]):
            img = img + torch.sigmoid(-sd / s)[..., None] * (pal[6] - img)
            img = img + torch.sigmoid((OUTLINE - sd.abs()) / s)[..., None] * (pal[5] - img)
        return img

    @staticmethod
    def loss(pred: torch.Tensor, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Robust per-candidate photometric error. pred (B, K, P, 3), obs (B, P, 3), w (B, P) -> (B, K)."""
        e = torch.sqrt(((pred - obs[:, None]) ** 2).sum(-1) + EPS ** 2) - EPS
        return (e * w[:, None]).sum(-1) / w.sum(-1, keepdim=True)


def _block_centroid(obs: torch.Tensor, pal: torch.Tensor, px, py, w) -> torch.Tensor:
    """Centroid (world px) of pixels whose colour is close to the block colours. obs (B, P, 3)."""
    d = torch.minimum((obs - pal[5]).norm(dim=-1), (obs - pal[6]).norm(dim=-1))
    m = (d < 0.12).float() * w
    n = m.sum(-1, keepdim=True)
    c = torch.stack([(m * px).sum(-1), (m * py).sum(-1)], -1) / n.clamp(min=1)
    return torch.where(n > 5, c, torch.full_like(c, 256.0))


CROP = 44   # output pixels; the T reaches at most 81 world px = 15.1 px from its area centroid


def fit_batch(ren: TeeRenderer, frames: np.ndarray, agent: np.ndarray, pal: torch.Tensor, s_final: float,
              iters: int = 120, n_starts: int = 8, s0: float = 10.0, crop: Optional[int] = CROP,
              init: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """frames uint8 (B, 96, 96, 3), agent (B, 2) world px -> pose (B, 3), loss (B,), margin (B,).
    With `crop`, only a crop x crop window around the block-coloured pixels is compared (4.8x faster)."""
    dev = ren.device
    b = len(frames)
    obs_img = torch.from_numpy(np.array(frames)).to(dev).float() / 255.0                # (B, 96, 96, 3)
    ag = torch.from_numpy(np.asarray(agent, np.float32)).to(dev)
    with torch.no_grad():
        px, py, w = ren.full_grid(b)
        obs = obs_img.reshape(b, -1, 3)
        c = _block_centroid(obs, pal, px, py, w)                                         # (B, 2)
        if crop is not None:
            obs, px, py, w = ren.crop(obs_img, c, crop)
        base = ren.base(ag, pal, s_final, px, py)
        if init is None:
            th = torch.arange(n_starts, device=dev, dtype=torch.float32) * (2 * math.pi / n_starts)
            th = th[None].expand(b, -1)
            x = c[:, 0:1] + torch.sin(th) * T_CENTROID_Y
            y = c[:, 1:2] - torch.cos(th) * T_CENTROID_Y
            p0 = torch.stack([x, y, th], -1)                                              # (B, K, 3)
        else:
            p0 = torch.from_numpy(np.asarray(init, np.float32)).to(dev)[:, None]
    pos = p0[..., :2].clone().requires_grad_(True)
    ang = p0[..., 2:].clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [pos], "lr": 2.0}, {"params": [ang], "lr": 0.04}])
    lr0 = [g["lr"] for g in opt.param_groups]
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        s = s_final + (s0 - s_final) * max(0.0, 1.0 - frac / 0.6)
        scale = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac))
        for g, l in zip(opt.param_groups, lr0):
            g["lr"] = l * scale
        loss = ren.loss(ren.with_tee(base, torch.cat([pos, ang], -1), pal, s, px, py), obs, w)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        pose = torch.cat([pos, ang], -1)
        loss = ren.loss(ren.with_tee(base, pose, pal, s_final, px, py), obs, w)        # (B, K)
        best = loss.argmin(1)
        bi = torch.arange(len(frames), device=dev)
        bp = pose[bi, best]
        # margin: how much worse the best start that converged to a *different* pose is (ambiguity measure)
        dpos = (pose[..., :2] - bp[:, None, :2]).norm(dim=-1)
        dang = torch.abs(torch.atan2(torch.sin(pose[..., 2] - bp[:, None, 2]), torch.cos(pose[..., 2] - bp[:, None, 2])))
        other = (dpos > 10) | (dang > math.radians(10))
        alt = torch.where(other, loss, torch.full_like(loss, float("inf"))).min(1).values
        margin = alt - loss[bi, best]
    out = bp.cpu().numpy().astype(np.float64)
    out[:, 2] = wrap_angle(out[:, 2])
    return {"pose": out, "loss": loss[bi, best].cpu().numpy(), "margin": margin.cpu().numpy()}


def calibrate(ren: TeeRenderer, frames: np.ndarray, agent: np.ndarray, n: int = 256, seed: int = 0,
              joint_iters: int = 300, log=print) -> Tuple[torch.Tensor, float]:
    """Learn the 7 colours and the edge blur `s` from the data itself (poses are unknown, fitted jointly)."""
    dev = ren.device
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(frames), size=min(n, len(frames)), replace=False))
    fr, ag = np.asarray(frames[idx]), np.asarray(agent[idx])
    bg = torch.from_numpy(np.median(fr.reshape(-1, 3), 0).astype(np.float32) / 255.0).to(dev)
    pal = torch.from_numpy(PALETTE0).to(dev) * bg / torch.ones(3, device=dev)            # rescale to observed white
    s = 2.0
    obs = torch.from_numpy(fr).to(dev).float().reshape(len(fr), -1, 3) / 255.0
    ag_t = torch.from_numpy(ag.astype(np.float32)).to(dev)
    px, py, w = ren.full_grid(len(fr))
    for rnd in range(2):
        poses = np.concatenate([fit_batch(ren, fr[i:i + 64], ag[i:i + 64], pal, s, crop=None)["pose"] for i in range(0, len(fr), 64)])
        p = torch.from_numpy(poses.astype(np.float32)).to(dev)[:, None].clone().requires_grad_(True)
        pal_p = pal.clone().requires_grad_(True)
        log_s = torch.tensor(math.log(s), device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [p], "lr": 0.3}, {"params": [pal_p], "lr": 0.003}, {"params": [log_s], "lr": 0.01}])
        for it in range(joint_iters):
            sb = log_s.exp()
            loss = ren.loss(ren.with_tee(ren.base(ag_t, pal_p, sb, px, py), p, pal_p, sb, px, py), obs, w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pal_p.clamp_(0, 1)
        pal, s = pal_p.detach(), float(log_s.exp())
        log(f"  calibration round {rnd + 1}: loss {loss.item():.4f}, blur s = {s:.2f} world px, colours (0-255):")
        for name, c in zip(PALETTE_NAMES, (pal * 255).round().int().tolist()):
            log(f"     {name:<12} {c}")
    return pal, s


def fit_all(ren: TeeRenderer, frames: np.ndarray, agent: np.ndarray, pal: torch.Tensor, s: float,
            batch: int = 128, log=print, **kw) -> Dict[str, np.ndarray]:
    import time
    res, t0 = {"pose": [], "loss": [], "margin": []}, time.perf_counter()
    for i in range(0, len(frames), batch):
        r = fit_batch(ren, np.asarray(frames[i:i + batch]), np.asarray(agent[i:i + batch]), pal, s, **kw)
        for k in res:
            res[k].append(r[k])
        if (i // batch) % 20 == 0:
            log(f"  fitted {min(i + batch, len(frames))}/{len(frames)} ({time.perf_counter() - t0:.0f}s)")
    return {k: np.concatenate(v) for k, v in res.items()}


# ------------------------------------------------------------------ geometry helpers (numpy / shapely)
def tee_polygon(pose) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return T_VERTS @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def coverage(poses: np.ndarray) -> np.ndarray:
    """Fraction of the goal T covered by the block T, exactly as gym_pusht computes it (shapely)."""
    import shapely
    goal = shapely.Polygon(tee_polygon(GOAL_POSE))
    polys = shapely.polygons(np.stack([tee_polygon(p) for p in poses]))
    return shapely.area(shapely.intersection(polys, goal)) / goal.area


def reward_mae(poses_next: np.ndarray, reward: np.ndarray, offset=(0.0, 0.0)) -> float:
    p = poses_next.copy()
    p[:, :2] += np.asarray(offset)
    return float(np.abs(np.clip(coverage(p) / 0.95, 0, 1) - reward).mean())


def offset_from_reward(pose: np.ndarray, reward: np.ndarray, episode: np.ndarray, min_reward: float = 0.2, log=print):
    """Constant world-frame correction of the fitted positions, estimated from the dataset itself.

    LeRobot computed next.reward(t) = clip(coverage(true pose(t+1)) / 0.95) from the TRUE pose, so the
    correction that makes the fitted poses reproduce the reward best is the offset between the image and
    world coordinates (pygame draws a shape at integer coordinate p into pixel [p, p+1), i.e. half a pixel
    off, plus codec chroma shifts). Coarse-to-fine grid search on frames with reward > min_reward
    (where coverage is sensitive to the pose)."""
    i = np.arange(len(pose) - 1)
    ok = (episode[i + 1] == episode[i]) & (reward[i] > min_reward)
    nxt, r = pose[i[ok] + 1], reward[i[ok]]
    best_off = np.zeros(2)
    best = before = reward_mae(nxt, r)
    for step in (0.2, 0.1, 0.05, 0.025):
        centre = best_off.copy()
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                off = centre + step * np.array([dx, dy])
                m = reward_mae(nxt, r, off)
                if m < best:
                    best, best_off = m, off
    log(f"  offset from reward ({len(r)} frames with reward > {min_reward}): ({best_off[0]:+.3f}, {best_off[1]:+.3f}) px, "
        f"|reward - predicted| {before:.5f} -> {best:.5f}")
    return best_off, before, best


def distance_agent_to_tee(poses: np.ndarray, agent: np.ndarray) -> np.ndarray:
    """Distance (world px) from the pusher centre to the block outline (0 if inside)."""
    import shapely
    polys = shapely.polygons(np.stack([tee_polygon(p) for p in poses]))
    return shapely.distance(polys, shapely.points(agent))

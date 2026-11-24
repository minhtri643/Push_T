"""Diffusion Transformer policy: denoises a 2 s future pusher path conditioned on S_t (+ agent_pos).

Epsilon-prediction DDPM training, 16-step DDIM sampling, and exact path <-> velocity conversion.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import normalize_pos, unnormalize_pos


# --------------------------------------------------------------------------- noise schedule
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=0.999).float()


class NoiseScheduler:
    def __init__(self, n_train_steps: int = 100):
        self.T = n_train_steps
        self.betas = cosine_beta_schedule(n_train_steps)
        self.alphas_bar = torch.cumprod(1.0 - self.betas, dim=0)  # [T], decreasing

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_bar.to(x0.device)[k][:, None, None]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def ddim_timesteps(self, n_infer: int = 16) -> list[int]:
        ts = np.linspace(self.T - 1, 0, n_infer).round().astype(int)
        return list(dict.fromkeys(int(t) for t in ts))  # descending, unique

    def ddim_step(self, x_k: torch.Tensor, eps_hat: torch.Tensor, k: int, k_prev: int) -> torch.Tensor:
        ab_all = self.alphas_bar.to(x_k.device)
        ab = ab_all[k]
        ab_prev = ab_all[k_prev] if k_prev >= 0 else torch.ones((), device=x_k.device)
        x0_hat = ((x_k - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt()).clamp(-1.0, 1.0)
        return ab_prev.sqrt() * x0_hat + (1.0 - ab_prev).sqrt() * eps_hat


# --------------------------------------------------------------------------- DiT
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def timestep_embedding(k: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=k.device, dtype=torch.float32) / half)
    args = k.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_ratio * d, d)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sh1, sc1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None, :] * a
        return x + g2[:, None, :] * self.mlp(modulate(self.norm2(x), sh2, sc2))


class DiT(nn.Module):
    def __init__(self, horizon: int = 20, d: int = 128, depth: int = 4, n_heads: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.1, cond_in: int = 258):
        super().__init__()
        self.d = d
        self.path_embed = nn.Linear(2, d)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_embed = nn.Sequential(nn.Linear(cond_in, 256), nn.SiLU(), nn.Linear(256, d))
        self.blocks = nn.ModuleList([DiTBlock(d, n_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_linear = nn.Linear(d, 2)
        for m in (self.final_ada[1], self.final_linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, k: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B, H, 2] noisy path, k [B] int timestep, cond [B, cond_in] -> eps_hat [B, H, 2]."""
        c = self.t_embed(timestep_embedding(k, self.d)) + self.cond_embed(cond)
        h = self.path_embed(x) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.final_linear(modulate(self.final_norm(h), shift, scale))


# --------------------------------------------------------------------------- policy wrapper
class DiffusionPolicy(nn.Module):
    def __init__(self, pcfg: dict, s_dim: int = 256, ws: float = 512.0):
        super().__init__()
        assert pcfg["prediction_type"] == "epsilon" and pcfg["beta_schedule"] == "cosine"
        self.H = int(pcfg["horizon"])
        self.use_proprio = bool(pcfg["use_proprio"])
        self.ws = float(ws)
        self.scheduler = NoiseScheduler(int(pcfg["n_diffusion_steps"]))
        self.dit = DiT(self.H, int(pcfg["d_model"]), int(pcfg["depth"]), int(pcfg["n_heads"]),
                       int(pcfg["mlp_ratio"]), float(pcfg["dropout"]),
                       cond_in=s_dim + (2 if self.use_proprio else 0))

    def _cond(self, s_t: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        if self.use_proprio:
            return torch.cat([s_t, normalize_pos(agent_pos, self.ws)], dim=-1)
        return s_t

    def compute_loss(self, s_t, agent_pos, path) -> torch.Tensor:
        """s_t [B, 256], agent_pos [B, 2] px, path [B, H, 2] px -> scalar MSE on epsilon."""
        x0 = normalize_pos(path, self.ws)
        noise = torch.randn_like(x0)
        k = torch.randint(0, self.scheduler.T, (x0.shape[0],), device=x0.device)
        eps_hat = self.dit(self.scheduler.add_noise(x0, noise, k), k, self._cond(s_t, agent_pos))
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample(self, s_t, agent_pos, n_infer: int = 16, generator: torch.Generator | None = None) -> torch.Tensor:
        """-> path_hat [B, H, 2] px (clamped to the workspace)."""
        B, dev = s_t.shape[0], s_t.device
        x = torch.randn(B, self.H, 2, device=dev, generator=generator)
        cond = self._cond(s_t, agent_pos)
        ts = self.scheduler.ddim_timesteps(n_infer)
        for i, k in enumerate(ts):
            k_prev = ts[i + 1] if i + 1 < len(ts) else -1
            eps = self.dit(x, torch.full((B,), k, device=dev, dtype=torch.long), cond)
            x = self.scheduler.ddim_step(x, eps, k, k_prev)
        return unnormalize_pos(x.clamp(-1.0, 1.0), self.ws)


# --------------------------------------------------------------------------- action conversion
def path_to_velocity(path: torch.Tensor, start_pos: torch.Tensor, dt: float) -> torch.Tensor:
    """path [B, H, 2] px, start_pos [B, 2] px -> velocity [B, H, 2] px/s."""
    prev = torch.cat([start_pos[:, None, :], path[:, :-1, :]], dim=1)
    return (path - prev) / dt


def velocity_to_targets(velocity: torch.Tensor, start_pos: torch.Tensor, dt: float) -> torch.Tensor:
    """velocity [B, H, 2] px/s -> absolute target positions [B, H, 2] px (the sim action)."""
    return start_pos[:, None, :] + torch.cumsum(velocity, dim=1) * dt


# --------------------------------------------------------------------------- dummy test
if __name__ == "__main__":
    from src.utils import get_device, load_config, set_seed

    set_seed(0)
    device = get_device()
    pcfg = load_config("configs/default.yaml")["policy"]
    pol = DiffusionPolicy(pcfg).to(device)
    n_par = sum(p.numel() for p in pol.parameters())
    print(f"device: {device} | DiT parameters: {n_par:,}")

    ab = pol.scheduler.alphas_bar
    assert bool((ab[1:] < ab[:-1]).all()), "alphas_bar must decrease monotonically"
    print(f"alphas_bar: first {ab[0]:.5f}, last {ab[-1]:.2e}")
    assert ab[0] > 0.999 and ab[-1] < 1e-2
    print("DDIM timesteps:", pol.scheduler.ddim_timesteps(16))

    B, H = 4, pol.H
    cond_in = 256 + (2 if pol.use_proprio else 0)
    x = torch.randn(B, H, 2, device=device)
    out0 = pol.dit(x, torch.randint(0, 100, (B,), device=device), torch.randn(B, cond_in, device=device))
    print("DiT forward:", tuple(out0.shape), "| max |output| right after init:", float(out0.abs().max()))
    assert out0.shape == (B, H, 2) and float(out0.abs().max()) == 0.0

    s_t = torch.randn(B, 256, device=device)
    pos = torch.rand(B, 2, device=device) * 512
    path = torch.rand(B, H, 2, device=device) * 512
    loss = pol.compute_loss(s_t, pos, path)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"compute_loss: {float(loss):.4f} (about 1.0 at init) | backward OK")

    # gradient reaches the condition MLP after a few updates (adaLN-Zero starts with zero grads upstream)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        pol.compute_loss(s_t, pos, path).backward()
        opt.step()
    opt.zero_grad()
    pol.compute_loss(s_t, pos, path).backward()
    g = float(pol.dit.cond_embed[0].weight.grad.abs().sum())
    print(f"cond MLP gradient |sum| after 5 updates: {g:.3e}")
    assert g > 0, "condition is not wired into the loss"

    pol.eval()
    sample = pol.sample(s_t, pos, n_infer=16)
    print("sample:", tuple(sample.shape), "range", float(sample.min()), float(sample.max()))
    assert sample.shape == (B, H, 2) and torch.isfinite(sample).all()
    assert float(sample.min()) >= 0 and float(sample.max()) <= 512

    v = path_to_velocity(path, pos, 0.1)
    rt = float((velocity_to_targets(v, pos, 0.1) - path).abs().max())
    print(f"round-trip max |error| = {rt:.2e} (expected < 1e-3)")
    assert rt < 1e-3

    # mini learnability check: overfit one fixed batch
    set_seed(1)
    pol2 = DiffusionPolicy({**pcfg, "dropout": 0.0}).to(device)
    opt = torch.optim.AdamW(pol2.parameters(), lr=1e-3)
    first = None
    for it in range(300):
        opt.zero_grad()
        l = pol2.compute_loss(s_t, pos, path)
        l.backward()
        opt.step()
        first = float(l) if first is None else first
    print(f"mini overfit: loss {first:.3f} -> {float(l):.3f} (should drop clearly)")
    assert float(l) < 0.8 * first
    print("dit_policy dummy test passed")

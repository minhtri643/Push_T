"""JEPA latent world model for PushT (96x96 RGB).

z_t = f(o_t) -> S_t = g([z_{t-1}, z_t, z_t - z_{t-1}]) -> latent rollout with actions -> predicted
EMA-target embeddings of future frames. Loss is computed in latent space only (no pixels).
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- helpers
def preprocess(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [..., H, W, 3] -> float [..., 3, H, W] normalised to [-1, 1]."""
    x = frames.float().div_(255.0).movedim(-1, -3)
    return (x - 0.5) / 0.5


def random_shift(frames: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """frames [B, T, C, H, W]: replicate-pad then random crop. Same offset for all T frames of a
    sample, different offsets across samples."""
    if pad <= 0:
        return frames
    B, T, C, H, W = frames.shape
    x = F.pad(frames.reshape(B, T * C, H, W), (pad,) * 4, mode="replicate")
    dev = x.device
    oy = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    out = torch.empty(B, T * C, H, W, dtype=x.dtype, device=dev)
    for i in range(B):
        out[i] = x[i, :, oy[i]:oy[i] + H, ox[i]:ox[i] + W]
    return out.reshape(B, T, C, H, W)


def vicreg_var(x: torch.Tensor) -> torch.Tensor:
    """x [N, D]: hinge on per-dimension std (anti-collapse)."""
    std = torch.sqrt(x.float().var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def vicreg_cov(x: torch.Tensor) -> torch.Tensor:
    """x [N, D]: squared off-diagonal covariance, decorrelates dimensions."""
    x = x.float()
    n, d = x.shape
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / d


@torch.no_grad()
def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of normalised singular values) of the mean-centred matrix x [N, D]."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    p = sv / sv.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------- modules
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


class FrameEncoder(nn.Module):
    """[B, 3, 96, 96] -> [B, z_dim]. Flatten (not GAP) keeps spatial position information."""

    def __init__(self, channels=(32, 64, 128, 256), z_dim: int = 128, img_size: int = 96):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, channels[0], 3, padding=1), _gn(channels[0]), nn.ReLU(inplace=True)]
        c_in = channels[0]
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, 3, stride=2, padding=1), _gn(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1), _gn(c), nn.ReLU(inplace=True),
            ]
            c_in = c
        layers += [nn.Conv2d(c_in, 32, 1), nn.Flatten()]
        self.conv = nn.Sequential(*layers)
        side = img_size // (2 ** len(channels))  # 96 -> 6
        self.fc = nn.Linear(32 * side * side, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ContextAggregator(nn.Module):
    """[B, 2, z] (frames t-1, t) -> S_t [B, s]."""

    def __init__(self, z_dim: int = 128, s_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * z_dim, s_dim), nn.GELU(), nn.Linear(s_dim, s_dim), nn.LayerNorm(s_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zp, zc = z[:, 0], z[:, 1]
        return self.net(torch.cat([zp, zc, zc - zp], dim=-1))


class ActionEmbed(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class LatentDynamics(nn.Module):
    """Residual step: s_k = s_{k-1} + F([s_{k-1}, e(a)])."""

    def __init__(self, s_dim: int = 256, act_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + act_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, s_dim),
        )

    def forward(self, s: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        return s + self.net(torch.cat([s, a_emb], dim=-1))


class Readout(nn.Module):
    def __init__(self, s_dim: int = 256, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, s_dim), nn.GELU(), nn.Linear(s_dim, z_dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# --------------------------------------------------------------------------- full model
class JEPAWorldModel(nn.Module):
    def __init__(self, wm_cfg: dict):
        super().__init__()
        self.z_dim, self.s_dim = int(wm_cfg["z_dim"]), int(wm_cfg["s_dim"])
        self.lambda_var = float(wm_cfg["lambda_var"])
        self.lambda_cov = float(wm_cfg["lambda_cov"])
        self.encoder = FrameEncoder(wm_cfg["enc_channels"], self.z_dim)
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_encoder.requires_grad_(False)
        self.aggregator = ContextAggregator(self.z_dim, self.s_dim)
        self.action_embed = ActionEmbed(int(wm_cfg["act_emb_dim"]))
        self.dynamics = LatentDynamics(self.s_dim, int(wm_cfg["act_emb_dim"]), int(wm_cfg["dyn_hidden"]))
        self.readout = Readout(self.s_dim, self.z_dim)

    # -- parameters -----------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        for p_ema, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            p_ema.mul_(tau).add_(p.detach(), alpha=1.0 - tau)

    # -- encoding -------------------------------------------------------------------
    def encode_frames(self, frames: torch.Tensor, target: bool = False) -> torch.Tensor:
        """[B, T, 96, 96, 3] uint8 (or already preprocessed float [B, T, 3, 96, 96]) -> [B, T, z]."""
        if frames.dtype == torch.uint8:
            frames = preprocess(frames)
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        if target:
            with torch.no_grad():  # chunked to keep peak VRAM low on a 4 GB GPU
                z = torch.cat([self.ema_encoder(x[i:i + 256]) for i in range(0, len(x), 256)])
        else:
            z = self.encoder(x)
        return z.reshape(B, T, -1)

    def encode_state(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, 2, 96, 96, 3] uint8 -> S_t [B, 256] (used by Step 4 and 5)."""
        return self.aggregator(self.encode_frames(frames))

    def rollout(self, s_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """s_t [B, 256], actions [B, K, 2] (normalised) -> predicted z [B, K, 128]."""
        a_emb = self.action_embed(actions)
        s, outs = s_t, []
        for k in range(actions.shape[1]):
            s = self.dynamics(s, a_emb[:, k])
            outs.append(s)
        return self.readout(torch.stack(outs, dim=1))

    # -- loss -----------------------------------------------------------------------
    def compute_loss(self, batch: dict) -> dict:
        """batch: obs [B,2,...], future_obs [B,K,...], actions [B,K,2], future_mask [B,K] bool.
        Frames may be uint8 (HWC) or preprocessed float (CHW)."""
        z_ctx = self.encode_frames(batch["obs"])                       # [B, 2, z]
        S = self.aggregator(z_ctx)                                     # [B, s]
        z_bar = F.layer_norm(self.encode_frames(batch["future_obs"], target=True), (self.z_dim,))
        z_hat = self.rollout(S, batch["actions"])                      # [B, K, z]

        m = batch["future_mask"].float()                               # [B, K]
        err = (z_hat - z_bar).abs().mean(dim=-1)                       # [B, K]
        l_pred = (err * m).sum() / m.sum().clamp_min(1.0)
        l_pred_k = (err * m).sum(0) / m.sum(0).clamp_min(1.0)          # [K]

        l_var = vicreg_var(S) + vicreg_var(z_ctx.reshape(-1, self.z_dim))
        l_cov = vicreg_cov(S)
        loss = l_pred + self.lambda_var * l_var + self.lambda_cov * l_cov
        return {
            "loss": loss, "l_pred": l_pred, "l_var": l_var, "l_cov": l_cov,
            "std_s": S.detach().std(dim=0).mean(),
            "std_z": z_ctx.detach().reshape(-1, self.z_dim).std(dim=0).mean(),
            "l_pred_k": l_pred_k.detach(),
        }


# --------------------------------------------------------------------------- dummy test
if __name__ == "__main__":
    from src.utils import get_device, load_config, set_seed

    set_seed(0)
    device = get_device()
    wm_cfg = load_config("configs/default.yaml")["wm"]
    model = JEPAWorldModel(wm_cfg).to(device)
    print("device:", device)

    print("\n--- parameters ---")
    for name in ["encoder", "aggregator", "action_embed", "dynamics", "readout"]:
        print(f"{name:13s}: {count_params(getattr(model, name)):>10,d}")
    print(f"{'TOTAL (trainable)':17s}: {count_params(model, trainable_only=True):>10,d}   (EMA copy not counted)")

    B, K = 4, int(wm_cfg["k_future"])
    frames = torch.randint(0, 255, (B, 2, 96, 96, 3), dtype=torch.uint8, device=device)
    S = model.encode_state(frames)
    print("\nencode_state:", tuple(S.shape))
    assert S.shape == (B, 256)
    z_hat = model.rollout(S, torch.randn(B, K, 2, device=device))
    print("rollout     :", tuple(z_hat.shape))
    assert z_hat.shape == (B, K, 128)

    batch = {
        "obs": torch.randint(0, 255, (8, 2, 96, 96, 3), dtype=torch.uint8, device=device),
        "future_obs": torch.randint(0, 255, (8, K, 96, 96, 3), dtype=torch.uint8, device=device),
        "actions": torch.randn(8, K, 2, device=device),
        "future_mask": torch.ones(8, K, dtype=torch.bool, device=device),
    }
    batch["future_mask"][:, -2:] = False
    out = model.compute_loss(batch)
    print("losses:", {k: round(float(v), 4) for k, v in out.items() if k != "l_pred_k"})
    assert all(torch.isfinite(v).all() for v in out.values())
    out["loss"].backward()
    assert all(p.grad is None for p in model.ema_encoder.parameters()), "EMA encoder must have no grad"
    assert any(p.grad is not None for p in model.encoder.parameters()), "online encoder must have grad"
    print("backward OK; EMA encoder has no grad")

    # random_shift: identical offset across the T frames of one sample
    x = torch.randn(3, 1, 3, 96, 96, device=device).expand(3, 5, 3, 96, 96).contiguous()
    y = random_shift(x, 4)
    assert y.shape == x.shape and torch.allclose(y[:, 0], y[:, 4]), "offset must be shared across frames"
    assert torch.equal(random_shift(x, 0), x)
    print("random_shift OK")

    # EMA moves towards online
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
    d0 = sum((a - b).abs().sum() for a, b in zip(model.ema_encoder.parameters(), model.encoder.parameters()))
    model.update_ema(0.9)
    d1 = sum((a - b).abs().sum() for a, b in zip(model.ema_encoder.parameters(), model.encoder.parameters()))
    print(f"EMA gap shrinks by factor {float(d1 / d0):.3f} (expected 0.900)")
    print("effective_rank(randn 1000x256) =", round(effective_rank(torch.randn(1000, 256)), 1), "(should be close to 256)")

    # timing + memory for a training-size batch (B=64, K=8) through the real training path
    model.zero_grad()
    Bt = int(wm_cfg["batch_size"])
    big = {
        "obs": torch.randint(0, 255, (Bt, 2, 96, 96, 3), dtype=torch.uint8, device=device),
        "future_obs": torch.randint(0, 255, (Bt, K, 96, 96, 3), dtype=torch.uint8, device=device),
        "actions": torch.randn(Bt, K, 2, device=device),
        "future_mask": torch.ones(Bt, K, dtype=torch.bool, device=device),
    }

    def train_like_step():
        allf = random_shift(preprocess(torch.cat([big["obs"], big["future_obs"]], dim=1)), 4)
        b = dict(big, obs=allf[:, :2], future_obs=allf[:, 2:])
        model.zero_grad()
        model.compute_loss(b)["loss"].backward()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        train_like_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        train_like_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    iters = 25650 * 0.9 / Bt
    print(f"\nB={Bt}, K={K}: {ms:.0f} ms / iteration -> about {ms * iters / 1000 / 60:.1f} min / epoch (estimate)")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("world_model dummy test passed")

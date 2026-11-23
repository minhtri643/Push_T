"""Self-trained image encoder for the vision World Model (no pretrained weights, Push-T frames only).

Replaces the frozen DINOv2 encoder. It is trained JEPA-style together with the action-conditioned
predictor (scripts/6_vision_world_model/train_jepa_wm.py): the predictor must predict the encoder's features of the
future frame, the targets come from an EMA copy of the encoder (no gradient), and variance /
covariance regularisation (VICReg) plus an inverse-dynamics head keep the features from collapsing.
This is the recipe of PLDM (Sobal et al. 2025, "Learning from reward-free offline data: a case for
planning with latent dynamics models"), adapted to images of Push-T.

96x96 RGB -> 12x12 grid of tokens, `feat_dim` channels each. Two coordinate channels are added to the
input so the features can encode absolute position (the block pose is absolute).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 12


def _block(cin, cout, k, s, p):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p), nn.GroupNorm(8, cout), nn.GELU())


class ConvEncoder(nn.Module):
    def __init__(self, width: int = 32, feat_dim: int = 64):
        super().__init__()
        w = width
        # stride-2 stem: a full-resolution 96x96 layer made training ~5x slower on the 4 GB laptop GPU
        self.net = nn.Sequential(
            _block(5, w, 4, 2, 1),            # 48
            _block(w, 2 * w, 3, 1, 1),
            _block(2 * w, 2 * w, 4, 2, 1),    # 24
            _block(2 * w, 2 * w, 3, 1, 1),
            _block(2 * w, 4 * w, 4, 2, 1),    # 12
            _block(4 * w, 4 * w, 3, 1, 1),
            nn.Conv2d(4 * w, feat_dim, 1),
        )
        c = torch.linspace(-1, 1, 96)
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None], persistent=False)
        self.hparams = {"width": width, "feat_dim": feat_dim}
        self.n_tok, self.feat_dim = GRID * GRID, feat_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames uint8 (B, 96, 96, 3) or float in [0, 1] (B, 3, 96, 96) -> tokens (B, 144, feat_dim)."""
        if frames.dtype == torch.uint8:
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
        x = torch.cat([frames - 0.5, self.coords.expand(len(frames), -1, -1, -1)], 1)
        return self.net(x).flatten(2).transpose(1, 2)


def token_norm(z: torch.Tensor) -> torch.Tensor:
    """Parameter-free layer norm over channels of every token (used for predictor inputs and targets)."""
    return F.layer_norm(z.float(), z.shape[-1:])


class FrameEncoder:
    """Same interface as the DINOv2 wrapper used by the policy: .encode(uint8 frames) -> features."""

    def __init__(self, enc: ConvEncoder, device: str):
        self.enc, self.device = enc.eval(), device

    def encode(self, frames: np.ndarray) -> torch.Tensor:
        return encode_frames(self.enc, frames, self.device)


@torch.no_grad()
def encode_frames(enc: ConvEncoder, frames, device: str, batch_size: int = 256) -> torch.Tensor:
    """numpy/torch uint8 (N, 96, 96, 3) -> normalised tokens float16 on CPU (N, 144, feat_dim)."""
    out = []
    for i in range(0, len(frames), batch_size):
        f = frames[i:i + batch_size]
        f = torch.from_numpy(np.ascontiguousarray(f)) if isinstance(f, np.ndarray) else f
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            z = enc(f.to(device))
        out.append(token_norm(z).half().cpu())
    return torch.cat(out)

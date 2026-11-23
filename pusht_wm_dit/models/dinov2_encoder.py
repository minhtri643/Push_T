"""Frozen pretrained visual encoder (DINOv2 ViT-S/14) for the vision World Model.

Follows the recipe of recent world models (DINO-WM, V-JEPA 2-AC): keep a large self-supervised image
encoder frozen and learn only an action-conditioned predictor on top of its patch features.

Loaded from the local torch.hub cache only (no network): download it once with
    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"

A 96x96 Push-T frame is resized to 224x224 -> 16x16 patch tokens (384-d) -> average-pooled to
8x8 = 64 tokens, stored as float16 (49 KB per frame).
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
REPO_DIR = os.path.join(HUB_DIR, "facebookresearch_dinov2_main")
GRID = 8          # pooled patch grid (GRID x GRID tokens)
FEAT_DIM = 384    # ViT-S/14 width
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoEncoder:
    """frames uint8 (B, 96, 96, 3) -> patch features float16 (B, GRID*GRID, FEAT_DIM)."""

    def __init__(self, device: str = "cuda"):
        if not os.path.exists(os.path.join(REPO_DIR, "hubconf.py")):
            raise FileNotFoundError(f"DINOv2 not in the local torch.hub cache ({REPO_DIR}); see module docstring.")
        self.device = device
        self.model = torch.hub.load(REPO_DIR, "dinov2_vits14", source="local", verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(device), _STD.to(device)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 256, grid: int = GRID) -> torch.Tensor:
        """grid = 8 (pooled, 64 tokens) or 16 (full 224-px patch grid, 256 tokens)."""
        out = []
        for i in range(0, len(frames), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch_size])).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                tok = self.model.forward_features(x)["x_norm_patchtokens"]          # (B, 256, 384)
            tok = tok.float().view(-1, 16, 16, FEAT_DIM).permute(0, 3, 1, 2)
            if grid != 16:
                tok = F.avg_pool2d(tok, 16 // grid)                                 # (B, 384, grid, grid)
            out.append(tok.flatten(2).transpose(1, 2).half())                       # (B, grid*grid, 384)
        return torch.cat(out)


_shared: Optional[DinoEncoder] = None


def get_encoder(device: str = "cuda") -> DinoEncoder:
    global _shared
    if _shared is None:
        _shared = DinoEncoder(device)
    return _shared

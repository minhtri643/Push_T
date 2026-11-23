"""Frozen pretrained image/video encoders for the vision World Model ("standing on the shoulders of giants").

The encoder is FROZEN: no gradient, never fine-tuned. Only the action-conditioned predictor and the pose
probe on top of its patch features are trained, on lerobot/pusht only (scripts/6_vision_world_model/train_frozen_encoder_wm.py).
This is the recipe of V-JEPA 2-AC (Meta, 2025) and DINO-WM (2024).

    vjepa2   V-JEPA 2 ViT-L/16 (300M params, 256 px, self-supervised on >1M hours of video), Hugging Face
             facebook/vjepa2-vitl-fpc64-256, loaded with transformers.VJEPA2Model. Download once:
             python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/vjepa2-vitl-fpc64-256', allow_patterns=['*.json', 'model.safetensors'])"
    dinov2   DINOv2 ViT-S/14 (models/dinov2_encoder.py, local torch.hub cache)

A 96x96 Push-T frame is resized to the encoder resolution (whole frame, no crop: the walls are at the
border), normalised with the ImageNet statistics, and its 16x16 patch tokens are average-pooled to 8x8 = 64
tokens. V-JEPA 2 is a video model with tubelet size 2: a single frame is repeated to a 2-frame clip.
Features are returned as float16 on the CPU: (N, 64, feat_dim).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

GRID = 8
VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VJEPA2Encoder:
    """frames uint8 (B, 96, 96, 3) -> pooled patch features float16 on CPU (B, 64, 1024)."""

    name = "vjepa2"

    def __init__(self, device: str = "cuda", model_id: str = VJEPA2_ID):
        from transformers import VJEPA2Model

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        try:   # local cache first (no network round trips once downloaded)
            model = VJEPA2Model.from_pretrained(model_id, dtype=dtype, local_files_only=True)
        except OSError:
            model = VJEPA2Model.from_pretrained(model_id, dtype=dtype)
        self.encoder = model.encoder.to(device).eval()      # the pretrained predictor is not needed
        del model
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        cfg = self.encoder.config
        self.size, self.patch = cfg.crop_size, cfg.patch_size
        self.side = self.size // self.patch                 # 16
        self.feat_dim, self.n_tok = cfg.hidden_size, GRID * GRID
        self.device = device
        self.mean, self.std = _MEAN.to(device), _STD.to(device)
        self.dtype = next(self.encoder.parameters()).dtype

    @torch.no_grad()
    def encode(self, frames, batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(frames), batch_size):
            f = frames[i:i + batch_size]
            x = torch.from_numpy(np.ascontiguousarray(f)) if isinstance(f, np.ndarray) else f
            x = x.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False).clamp(0, 1)
            x = ((x - self.mean) / self.std).to(self.dtype)
            clip = x[:, None].expand(-1, 2, -1, -1, -1)       # (B, T=2, C, H, W): one frame as a 2-frame clip
            tok = self.encoder(pixel_values_videos=clip).last_hidden_state.float()   # (B, 256, 1024)
            tok = tok.view(-1, self.side, self.side, self.feat_dim).permute(0, 3, 1, 2)
            tok = F.avg_pool2d(tok, self.side // GRID)                              # (B, 1024, 8, 8)
            out.append(tok.flatten(2).transpose(1, 2).half().cpu())
        return torch.cat(out)


class DinoV2Encoder:
    """Wrapper of the legacy DINOv2 encoder with the same interface (features float16 on CPU)."""

    name = "dinov2"

    def __init__(self, device: str = "cuda"):
        from pusht_wm_dit.models.dinov2_encoder import FEAT_DIM, get_encoder
        self.enc = get_encoder(device)
        self.feat_dim, self.n_tok, self.device = FEAT_DIM, GRID * GRID, device

    @torch.no_grad()
    def encode(self, frames, batch_size: int = 256) -> torch.Tensor:
        return self.enc.encode(frames, batch_size=batch_size).cpu()


_shared: Dict[str, object] = {}


def get_frozen_encoder(name: str, device: str = "cuda"):
    """name = 'vjepa2' | 'dinov2'. One shared instance per name."""
    if name not in _shared:
        if name == "vjepa2":
            _shared[name] = VJEPA2Encoder(device)
        elif name == "dinov2":
            _shared[name] = DinoV2Encoder(device)
        else:
            raise ValueError(f"unknown frozen encoder {name!r}")
    return _shared[name]

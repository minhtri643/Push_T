"""Vision World Model: action-conditioned predictor in the latent space of an image encoder.

    encoder features z_t  +  robot actions  ->  predicted features z_{t+S}

Two encoders are supported:
  * self-trained ConvEncoder (models/jepa_encoder.py, 12 x 12 tokens x 64), trained jointly with this
    predictor JEPA-style on lerobot/pusht only (scripts/6_vision_world_model/train_jepa_wm.py) - the current default;
  * legacy: frozen pretrained DINOv2 patch features (8 x 8 x 384, DINO-WM style, scripts/09).

The model never sees the block pose. It predicts the next *image features* (S = 4 control steps,
0.4 s, ahead) from the two latest frames, the pusher position (proprioception) and the S actions,
and can be rolled out autoregressively.

A separate PoseProbe reads the T-block pose from features. It is trained afterwards, only to
measure how good the imagination is in pixels/degrees and to visualise it; the World Model itself
is trained on features alone.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pusht_wm_dit.models.dinov2_encoder import FEAT_DIM, GRID

N_TOK = GRID * GRID


class FeatureNorm(nn.Module):
    """Per-channel standardisation of encoder features (statistics from the training set)."""

    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.register_buffer("mean", mean if mean is not None else torch.zeros(feat_dim))
        self.register_buffer("std", std if std is not None else torch.ones(feat_dim))

    def forward(self, z):
        return (z.float() - self.mean) / self.std


class LatentPredictor(nn.Module):
    """(z_{t-S}, z_t, agent_{t-S}, agent_t, a_t..a_{t+S-1}) -> (z_{t+S}, agent_{t+S}); features normalised.
    n_tok / feat_dim default to the DINOv2 setting (64 x 384); the self-trained JEPA encoder uses 144 x 64."""

    def __init__(self, skip: int = 4, dim: int = 256, depth: int = 4, heads: int = 4, n_tok: int = N_TOK, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.skip, self.dim, self.n_tok = skip, dim, n_tok
        self.in_proj = nn.Linear(feat_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, 2 * n_tok + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        cond_in = 4 + 4 * skip  # two pusher positions, S absolute actions, S actions relative to the pusher
        self.cond = nn.Sequential(nn.Linear(cond_in, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, feat_dim)
        self.agent_head = nn.Linear(dim, 2)
        for h in (self.head, self.agent_head):  # start as "nothing changes"
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.hparams = {"skip": skip, "dim": dim, "depth": depth, "heads": heads, "n_tok": n_tok, "feat_dim": feat_dim}

    def step(self, z_prev, z_cur, ag_prev, ag_cur, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """z_*: (B, n_tok, feat_dim) normalised; ag_*: (B, 2); actions: (B, S, 2) normalised positions."""
        b = z_cur.shape[0]
        rel = actions - ag_cur[:, None]
        c = self.cond(torch.cat([ag_prev, ag_cur, actions.reshape(b, -1), rel.reshape(b, -1)], -1))
        tok = torch.cat([self.in_proj(z_prev), self.in_proj(z_cur) + c[:, None], c[:, None]], 1) + self.pos
        h = self.norm(self.blocks(tok))
        z_next = z_cur + self.head(h[:, self.n_tok:2 * self.n_tok])
        ag_next = ag_cur + self.agent_head(h[:, -1]) * 0.2
        return z_next, ag_next

    def rollout(self, z_prev, z_cur, ag_prev, ag_cur, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """actions (B, K*S, 2) -> features (B, K, 64, 384) and pusher positions (B, K, 2) at t+S, ..., t+K*S."""
        S = self.skip
        zs, ags = [], []
        for k in range(actions.shape[1] // S):
            z_next, ag_next = self.step(z_prev, z_cur, ag_prev, ag_cur, actions[:, k * S:(k + 1) * S])
            zs.append(z_next)
            ags.append(ag_next)
            z_prev, z_cur, ag_prev, ag_cur = z_cur, z_next, ag_cur, ag_next
        return torch.stack(zs, 1), torch.stack(ags, 1)


class PoseProbe(nn.Module):
    """Normalised features (B, n_tok, feat_dim) -> block pose [x, y, sin, cos] (normalised). Measurement only."""

    def __init__(self, hidden: int = 512, n_tok: int = N_TOK, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(feat_dim, 32), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(n_tok * 32, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, z):
        return self.mlp(self.reduce(z).flatten(1))


def pose4_to_xyth(p4: torch.Tensor) -> torch.Tensor:
    return torch.stack([p4[..., 0], p4[..., 1], torch.atan2(p4[..., 2], p4[..., 3])], -1)


def load_vision_wm(path: str, device: str) -> Dict[str, nn.Module]:
    """Loads a vision World Model checkpoint. Checkpoints with an "encoder" entry carry their own
    self-trained (JEPA) encoder; older ones expect frozen DINOv2 features."""
    ck = torch.load(path, map_location=device, weights_only=False)
    hp = ck["hparams"]
    n_tok, feat_dim = hp.get("n_tok", N_TOK), hp.get("feat_dim", FEAT_DIM)
    norm = FeatureNorm(feat_dim=feat_dim).to(device)
    norm.load_state_dict(ck["norm"])
    pred = LatentPredictor(**hp).to(device)
    pred.load_state_dict(ck["predictor"])
    probe = PoseProbe(n_tok=n_tok, feat_dim=feat_dim).to(device)
    probe.load_state_dict(ck["probe"])
    out = {"norm": norm, "predictor": pred, "probe": probe}
    if "encoder" in ck:
        from pusht_wm_dit.models.jepa_encoder import ConvEncoder
        enc = ConvEncoder(**ck["encoder_hparams"]).to(device)
        enc.load_state_dict(ck["encoder"])
        out["encoder"] = enc
    for m in out.values():
        m.eval()
    # frozen pretrained encoder named in the checkpoint (train_frozen_encoder_wm.py: "vjepa2" | "dinov2"); None for JEPA / legacy DINOv2
    out["encoder_type"] = ck.get("encoder_type")
    return out

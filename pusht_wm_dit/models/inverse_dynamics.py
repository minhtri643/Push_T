"""Inverse Dynamics Controller for Push-T Action Execution.

Maps (s_t, hat_path, agent_pos) -> low-level control action sequence [x_cmd, y_cmd] in [-1, 1]^2
for Receding Horizon Control (predict H=16, execute Ta=8).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMlpBlock(nn.Module):
    """Residual MLP Block with LayerNorm and GELU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class InverseDynamicsController(nn.Module):
    """Deep Residual Inverse Dynamics Controller with relative proprioceptive grounding."""

    def __init__(
        self,
        latent_dim: int = 128,
        horizon: int = 16,
        traj_dim: int = 4,
        action_dim: int = 2,
        exec_steps: int = 8,
        agent_pos_dim: int = 2,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.traj_dim = traj_dim
        self.action_dim = action_dim
        self.exec_steps = exec_steps
        self.agent_pos_dim = agent_pos_dim

        hidden_dim = hidden_dims[0] if len(hidden_dims) > 0 else 256

        # 1. Trajectory Encoder: encodes Hx4 trajectory points
        self.traj_encoder = nn.Sequential(
            nn.Linear(horizon * traj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. Input projection: s_t (128) + agent_pos (2) + rel_agent_block (2) + traj_feat (hidden_dim)
        in_dim = latent_dim + agent_pos_dim + 2 + hidden_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Residual backbone for complex contact dynamics
        self.res_blocks = nn.Sequential(
            ResMlpBlock(hidden_dim),
            ResMlpBlock(hidden_dim),
        )

        # 4. Action prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, exec_steps * action_dim),
            nn.Tanh(),  # Bounded commands in [-1, 1]
        )

    def forward(
        self,
        s_t: torch.Tensor,
        predicted_path: torch.Tensor,
        agent_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts action sequence for Receding Horizon Control.

        Args:
            s_t: Latent state from World Model (B, latent_dim)
            predicted_path: Denoised trajectory from DiT (B, H, 4)
            agent_pos: Current agent end-effector position (B, 2) in [-1, 1]

        Returns:
            Action commands of shape (B, exec_steps, action_dim) in [-1, 1]
        """
        batch_size = s_t.shape[0]
        if agent_pos is None:
            agent_pos = torch.zeros((batch_size, self.agent_pos_dim), device=s_t.device, dtype=s_t.dtype)

        # Compute relative vector from block origin to pusher
        curr_block_pos = predicted_path[:, 0, :2]
        rel_pos = agent_pos - curr_block_pos

        # Encode trajectory
        flat_path = predicted_path.reshape(batch_size, -1)
        traj_feat = self.traj_encoder(flat_path)

        # Fuse representations
        fused = torch.cat([s_t, agent_pos, rel_pos, traj_feat], dim=-1)
        feat = self.in_proj(fused)
        feat = self.res_blocks(feat)

        actions_flat = self.head(feat)
        actions = actions_flat.view(batch_size, self.exec_steps, self.action_dim)
        return actions

    def compute_loss(
        self,
        s_t: torch.Tensor,
        predicted_path: torch.Tensor,
        target_actions: torch.Tensor,
        agent_pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes supervised action imitation MSE loss."""
        pred_actions = self.forward(s_t, predicted_path, agent_pos=agent_pos)
        loss = F.mse_loss(pred_actions, target_actions)
        return {
            "loss": loss,
            "pred_actions": pred_actions,
        }

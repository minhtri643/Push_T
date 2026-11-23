"""Action-conditioned Push-T dynamics: (current state, robot action) -> change of the object pose.

This is the part of the World Model that the project brief asks for:
"Từ trạng thái hiện tại và hành động của robot, mô hình dự đoán sự thay đổi vị trí vật thể."

State of one time step (normalized like the dataset: positions in [-1, 1], angle in radians):
    block     = (bx, by, theta)        T-block pose
    agent     = (ax, ay)               pusher position
    agent_vel = agent_t - agent_{t-1}  pusher displacement over the last step
Action: target position of the pusher (normalized), exactly as in the dataset.

The network sees the action in the block's own frame, so what it learns about a push does not
depend on where the block lies or how it is rotated. Absolute position and heading are still
inputs, so effects of the walls can be learned. Outputs are deltas in the block frame. The last
layer starts at zero, so an untrained model equals the "nothing moves" baseline.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

POS_SCALE = 0.05    # block displacement per step, normalized units (~12.8 px)
ROT_SCALE = 0.10    # block rotation per step, radians (~5.7 deg)
AGENT_SCALE = 0.20  # pusher displacement per step, normalized units (~51 px)
REL_SCALE = 4.0     # relative geometry features are ~0.25 normalized -> ~1
VEL_SCALE = 10.0    # pusher velocity features


class PushDynamics(nn.Module):
    """MLP dynamics model operating in the T-block frame."""

    IN_DIM = 12
    OUT_DIM = 5  # block dx, dy, dtheta (block frame), pusher dx, dy (block frame)

    def __init__(self, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(self.IN_DIM, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        mods.append(nn.Linear(hidden, self.OUT_DIM))
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.hparams = {"hidden": hidden, "layers": layers}

    @staticmethod
    def _to_block_frame(vx, vy, c, s):
        return vx * c + vy * s, -vx * s + vy * c

    def features(self, block, agent, agent_vel, action) -> torch.Tensor:
        bx, by, th = block[..., 0], block[..., 1], block[..., 2]
        c, s = torch.cos(th), torch.sin(th)
        ar = self._to_block_frame(agent[..., 0] - bx, agent[..., 1] - by, c, s)
        cr = self._to_block_frame(action[..., 0] - bx, action[..., 1] - by, c, s)
        aa = self._to_block_frame(action[..., 0] - agent[..., 0], action[..., 1] - agent[..., 1], c, s)
        av = self._to_block_frame(agent_vel[..., 0], agent_vel[..., 1], c, s)
        return torch.stack(
            [
                bx, by, s, c,
                ar[0] * REL_SCALE, ar[1] * REL_SCALE,
                cr[0] * REL_SCALE, cr[1] * REL_SCALE,
                aa[0] * REL_SCALE, aa[1] * REL_SCALE,
                av[0] * VEL_SCALE, av[1] * VEL_SCALE,
            ],
            dim=-1,
        )

    def step(self, block, agent, agent_vel, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. Returns (next_block, next_agent, next_agent_vel)."""
        out = self.net(self.features(block, agent, agent_vel, action))
        th = block[..., 2]
        c, s = torch.cos(th), torch.sin(th)
        dxl, dyl = out[..., 0] * POS_SCALE, out[..., 1] * POS_SCALE
        dax_l, day_l = out[..., 3] * AGENT_SCALE, out[..., 4] * AGENT_SCALE
        next_block = torch.stack(
            [block[..., 0] + dxl * c - dyl * s, block[..., 1] + dxl * s + dyl * c, th + out[..., 2] * ROT_SCALE],
            dim=-1,
        )
        next_agent = torch.stack([agent[..., 0] + dax_l * c - day_l * s, agent[..., 1] + dax_l * s + day_l * c], dim=-1)
        return next_block, next_agent, next_agent - agent

    def rollout(self, block, agent, agent_vel, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop imagination. actions (B, K, 2) -> blocks (B, K, 3), agents (B, K, 2) for t+1..t+K."""
        blocks, agents = [], []
        for k in range(actions.shape[1]):
            block, agent, agent_vel = self.step(block, agent, agent_vel, actions[:, k])
            blocks.append(block)
            agents.append(agent)
        return torch.stack(blocks, 1), torch.stack(agents, 1)


def load_dynamics(path: str, device: str) -> PushDynamics:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PushDynamics(**ckpt.get("hparams", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def pose_errors(pred_block: torch.Tensor, true_block: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Position error in pixels and wrapped angle error in degrees (normalized inputs)."""
    pos = torch.linalg.norm(pred_block[..., :2] - true_block[..., :2], dim=-1) * 256.0
    d = pred_block[..., 2] - true_block[..., 2]
    ang = torch.rad2deg(torch.abs(torch.atan2(torch.sin(d), torch.cos(d))))
    return {"pos_px": pos, "ang_deg": ang}

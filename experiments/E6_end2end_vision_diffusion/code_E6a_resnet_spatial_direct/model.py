"""Model definition for Stage E6a:
End-to-End ResNet-18 + Spatial-Softmax + DiT Action Diffusion Policy.

Direct Action Diffusion from raw video frames (no Inverse Dynamics, no privileged block pose).
T_obs = 2 consecutive frames (I_{t-1}, I_t).
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GroupNormBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, num_groups: int = 16):
        super().__init__()
        g1 = min(num_groups, planes)
        while planes % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(num_groups, planes)
        while planes % g2 != 0 and g2 > 1:
            g2 -= 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_groups=g1, num_channels=planes)
        self.act = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups=g2, num_channels=planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class ResNet18FeatureExtractor(nn.Module):
    """Custom ResNet-18 with GroupNorm preserving 6x6 spatial resolution for 96x96 images."""

    def __init__(self, in_channels: int = 3, num_groups: int = 16):
        super().__init__()
        self.inplanes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 96 -> 48
        self.gn1 = nn.GroupNorm(num_groups=min(num_groups, 64), num_channels=64)
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 48 -> 24

        self.layer1 = self._make_layer(64, 2, stride=1)   # 24 -> 24
        self.layer2 = self._make_layer(128, 2, stride=2)  # 24 -> 12
        self.layer3 = self._make_layer(256, 2, stride=2)  # 12 -> 6
        # layer4 with stride 1 to maintain 6x6 spatial resolution:
        self.layer4 = self._make_layer(512, 2, stride=1)  # 6 -> 6

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            g_down = min(self.num_groups, planes)
            while planes % g_down != 0 and g_down > 1:
                g_down -= 1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.GroupNorm(num_groups=g_down, num_channels=planes),
            )
        layers = [GroupNormBlock(self.inplanes, planes, stride=stride, downsample=downsample, num_groups=self.num_groups)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(GroupNormBlock(self.inplanes, planes, num_groups=self.num_groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 96, 96) -> out: (B, 512, 6, 6)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SpatialSoftmax(nn.Module):
    """Spatial Softmax layer mapping feature maps (B, C, H, W) -> K keypoint coordinates (B, 2*K) in [-1, 1]."""

    def __init__(self, in_channels: int, num_keypoints: int = 16, temperature: float = 1.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        self.keypoint_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map: (B, C, H, W)
        B, C, H, W = feature_map.shape
        heatmap = self.keypoint_conv(feature_map)  # (B, K, H, W)
        heatmap = heatmap.view(B, self.num_keypoints, H * W)
        probs = F.softmax(heatmap / self.temperature, dim=-1)  # (B, K, H*W)

        # Generate normalized coordinate grids in [-1, 1]
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=feature_map.device, dtype=feature_map.dtype),
            torch.linspace(-1.0, 1.0, W, device=feature_map.device, dtype=feature_map.dtype),
            indexing="ij",
        )
        pos_x = pos_x.reshape(1, 1, H * W)  # (1, 1, H*W)
        pos_y = pos_y.reshape(1, 1, H * W)  # (1, 1, H*W)

        # Expected coordinates
        expected_x = torch.sum(probs * pos_x, dim=-1)  # (B, K)
        expected_y = torch.sum(probs * pos_y, dim=-1)  # (B, K)

        keypoints = torch.stack([expected_x, expected_y], dim=-1)  # (B, K, 2)
        return keypoints.view(B, self.num_keypoints * 2)  # (B, 2*K)


class ResNetSpatialDirectPolicy(nn.Module):
    """End-to-End Direct Action Diffusion Policy:
    ResNet-18 (GroupNorm) + Spatial-Softmax + DiT Action Planner.

    Conditions on T_obs = 2 frames (I_{t-1}, I_t) and agent positions (p_{t-1}, p_t).
    Generates action sequence a_{t:t+H} directly without Inverse Dynamics.
    """

    def __init__(
        self,
        num_keypoints: int = 16,
        horizon: int = 16,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        diffusion_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.horizon = horizon

        # Shared vision backbone across observation timesteps
        self.backbone = ResNet18FeatureExtractor(in_channels=3, num_groups=16)
        self.spatial = SpatialSoftmax(in_channels=512, num_keypoints=num_keypoints)

        # Conditioning dimension: 2 frames * (2 * num_keypoints) + 2 frames * 2 (agent_pos)
        self.cond_dim = 2 * (num_keypoints * 2) + 2 * 2  # e.g., 2*(32) + 4 = 68

        # DiT Planner predicting continuous actions (ax, ay)
        self.dit = DiTTrajectoryPlanner(
            horizon=horizon,
            traj_dim=2,  # action dim = 2 (ax, ay)
            latent_dim=self.cond_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            diffusion_steps=diffusion_steps,
            beta_schedule=beta_schedule,
            lambda_fde=0.0,  # pure MSE action loss
        )

    def extract_features(self, imgs: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 2, 3, 96, 96) normalized [-1, 1] or ImageNet stats
        agent_pos: (B, 2, 2) normalized [-1, 1]
        Returns:
            cond: (B, cond_dim)
        """
        B, T_obs, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * T_obs, C, H, W)
        feat_maps = self.backbone(imgs_flat)  # (B*T, 512, 6, 6)
        kps = self.spatial(feat_maps)         # (B*T, 2*K)
        kps = kps.view(B, T_obs * self.num_keypoints * 2)  # (B, 2 * 2*K)

        agent_flat = agent_pos.view(B, T_obs * 2)          # (B, 4)
        cond = torch.cat([kps, agent_flat], dim=-1)        # (B, cond_dim)
        return cond

    def forward(
        self,
        imgs: torch.Tensor,
        agent_pos: torch.Tensor,
        actions_future: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass computing action diffusion MSE loss."""
        cond = self.extract_features(imgs, agent_pos)
        out = self.dit(actions_future, cond)
        return out

    @torch.no_grad()
    def sample_actions(
        self,
        imgs: torch.Tensor,
        agent_pos: torch.Tensor,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """Sampling predicted action chunk: (B, horizon, 2)."""
        cond = self.extract_features(imgs, agent_pos)
        return self.dit.sample(cond, num_inference_steps=num_inference_steps)

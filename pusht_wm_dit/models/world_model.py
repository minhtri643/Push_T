"""World Model for Push-T: ResNet-18 (GroupNorm) Encoder + Latent Dynamics.

Features:
1. ResNet-18 backbone with GroupNorm (replacing BatchNorm2d) for batch-independent representation stability.
2. Agent end-effector position fusion with RGB visual features into a compact latent state vector s_t in R^d.
3. Physical pose and agent coordinate decoders to ground the latent space.
4. Latent Transition Predictor (Forward Dynamics) predicting s_{t+1} = s_t + Delta s_t.
5. Weight freezing mechanism for downstream Diffusion training.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class GroupNormBlock(nn.Module):
    """ResNet BasicBlock utilizing GroupNorm instead of BatchNorm2d."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        num_groups: int = 16,
    ):
        super().__init__()
        # Determine valid num_groups for inplanes and planes
        g1 = min(num_groups, planes)
        while planes % g1 != 0 and g1 > 1:
            g1 -= 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_groups=g1, num_channels=planes)
        self.act = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups=g1, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet18GroupNorm(nn.Module):
    """Custom ResNet-18 vision encoder equipped with GroupNorm."""

    def __init__(self, in_channels: int = 3, num_groups: int = 16):
        super().__init__()
        self.inplanes = 64
        self.num_groups = num_groups

        # Initial convolution suitable for 96x96 RGB frames
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=min(num_groups, 64), num_channels=64)
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

        layers = []
        layers.append(
            GroupNormBlock(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                num_groups=self.num_groups,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                GroupNormBlock(
                    self.inplanes,
                    planes,
                    num_groups=self.num_groups,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W) -> (B, 3, 96, 96)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)
        return x


class LatentTransitionPredictor(nn.Module):
    """Predicts forward latent transition Delta s_t = T(s_t, a_t)."""

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, s_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts s_{t+1} from current latent state and action."""
        inp = torch.cat([s_t, action], dim=-1)
        delta = self.mlp(inp)
        return s_t + delta


class WorldModel(nn.Module):
    """Complete World Model combining ResNet-18 (GroupNorm), Pose Decoders, and Transition Predictor."""

    def __init__(
        self,
        latent_dim: int = 128,
        agent_pos_dim: int = 2,
        pose_dim: int = 4,
        action_dim: int = 2,
        num_groups: int = 16,
        transition_hidden_dim: int = 256,
        lambda_recon: float = 1.0,
        lambda_agent: float = 1.0,
        lambda_transition: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_recon = lambda_recon
        self.lambda_agent = lambda_agent
        self.lambda_transition = lambda_transition

        # 1. Vision Encoder
        self.vision_encoder = ResNet18GroupNorm(in_channels=3, num_groups=num_groups)

        # 2. Agent position encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_pos_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # 3. Fusion projection to compact latent state s_t
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 4. Decoders for physical grounding
        self.pose_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, pose_dim),
        )
        self.agent_pos_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, agent_pos_dim),
        )

        # 5. Latent Transition Dynamics Predictor
        self.transition_predictor = LatentTransitionPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=transition_hidden_dim,
        )

    def encode(self, frame: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """Encodes frame observation and agent position into compact latent state s_t."""
        vis_feat = self.vision_encoder(frame)      # (B, 512)
        ag_feat = self.agent_encoder(agent_pos)    # (B, 64)
        fused = torch.cat([vis_feat, ag_feat], dim=-1)
        s_t = self.fusion(fused)                   # (B, latent_dim)
        return s_t

    def decode(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes latent state into estimated 4D T-block pose and agent position."""
        pred_pose_4d = self.pose_decoder(s_t)
        pred_agent_pos = self.agent_pos_decoder(s_t)
        return pred_pose_4d, pred_agent_pos

    def predict_next_state(self, s_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts next latent state s_{t+1} = T(s_t, a_t)."""
        return self.transition_predictor(s_t, action)

    def compute_loss(
        self,
        frame: torch.Tensor,
        agent_pos: torch.Tensor,
        block_pose_4d: torch.Tensor,
        action: torch.Tensor,
        next_frame: torch.Tensor,
        next_agent_pos: torch.Tensor,
        next_block_pose_4d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Computes comprehensive World Model training loss."""
        # Current step representation
        s_t = self.encode(frame, agent_pos)
        pred_pose, pred_agent = self.decode(s_t)

        # Target step representation (stop gradient on target to prevent collapse)
        with torch.no_grad():
            target_s_next = self.encode(next_frame, next_agent_pos)

        # Predicted transition
        pred_s_next = self.predict_next_state(s_t, action)

        # Losses
        loss_pose = F.mse_loss(pred_pose, block_pose_4d)
        loss_agent = F.mse_loss(pred_agent, agent_pos)
        loss_transition = F.mse_loss(pred_s_next, target_s_next)

        total_loss = (
            self.lambda_recon * loss_pose
            + self.lambda_agent * loss_agent
            + self.lambda_transition * loss_transition
        )

        return {
            "total_loss": total_loss,
            "loss_pose": loss_pose,
            "loss_agent": loss_agent,
            "loss_transition": loss_transition,
            "s_t": s_t,
        }

    def freeze(self):
        """Freezes all model parameters and switches to eval mode for downstream DiT training."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print("World Model weights frozen successfully for Diffusion training.")

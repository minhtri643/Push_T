"""1D Diffusion Transformer (DiT) Trajectory Planner for Push-T.

Features:
1. 1D Transformer processing continuous 4D trajectories: [x_T, y_T, sin(theta), cos(theta)] in [-1, 1]^4 over horizon H=16.
2. Adaptive LayerNorm (AdaLN-Zero) conditioning on World Model latent state vector s_t and diffusion timestep k.
3. Cosine/linear variance schedule with both forward diffusion and fast DDIM reverse sampling.
4. Total training loss combining MSE trajectory reconstruction and auxiliary FDE^2 penalty:
   L_total = ||path_future - hat_path||_2^2 + lambda_fde * FDE^2.
"""

from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pusht_wm_dit.utils.metrics import compute_fde_torch


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulates LayerNorm output via AdaLN scale and shift: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock1D(nn.Module):
    """1D Transformer block with Adaptive LayerNorm (AdaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # AdaLN projection produces (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Initialize AdaLN modulation with zeros
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c is condition vector of shape (B, hidden_size)
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Multi-Head Self-Attention with AdaLN modulation and gating
        norm_x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Feed-Forward Network with AdaLN modulation and gating
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer1D(nn.Module):
    """Final AdaLN layer projecting hidden tokens back to trajectory dimension."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTTrajectoryPlanner(nn.Module):
    """1D Diffusion Transformer Planner for Push-T Trajectory Generation."""

    def __init__(
        self,
        horizon: int = 16,
        traj_dim: int = 4,
        latent_dim: int = 128,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        diffusion_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        w_xy: float = 1.0,
        w_theta: float = 1.0,
        lambda_fde: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.traj_dim = traj_dim
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.diffusion_steps = diffusion_steps
        self.w_xy = w_xy
        self.w_theta = w_theta
        self.lambda_fde = lambda_fde

        # 1. Trajectory Token & Position Embeddings
        self.token_embed = nn.Linear(traj_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 2. Diffusion Timestep Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # 3. Conditioning Fusion: Latent s_t + Timestep Embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim + model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # 4. DiT Transformer Blocks
        self.blocks = nn.ModuleList(
            [DiTBlock1D(model_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        # 5. Output Projection
        self.final_layer = FinalLayer1D(model_dim, traj_dim)

        # 6. Precompute Diffusion Noise Schedule
        self._register_diffusion_schedule(diffusion_steps, beta_schedule)

    def _register_diffusion_schedule(self, num_timesteps: int, schedule_type: str):
        """Constructs forward diffusion variance schedule."""
        if schedule_type == "squaredcos_cap_v2":
            # Cosine schedule (Nichol & Dhariwal)
            betas = []
            for i in range(num_timesteps):
                t1 = i / num_timesteps
                t2 = (i + 1) / num_timesteps
                s = 0.008
                f1 = math.cos((t1 + s) / (1.0 + s) * math.pi * 0.5) ** 2
                f2 = math.cos((t2 + s) / (1.0 + s) * math.pi * 0.5) ** 2
                betas.append(min(1.0 - f2 / f1, 0.999))
            betas = torch.tensor(betas, dtype=torch.float32)
        else:
            # Linear schedule
            betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def forward_dit(
        self,
        z_k: torch.Tensor,
        timesteps: torch.Tensor,
        s_t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoises trajectory z_k conditioned on diffusion timestep k and WM latent s_t.

        Args:
            z_k: Noisy trajectory (B, H, 4)
            timesteps: Diffusion timestep indices (B,)
            s_t: Latent state from frozen World Model (B, latent_dim)

        Returns:
            Predicted denoised trajectory hat_x_0 of shape (B, H, 4) in [-1, 1]^4
        """
        # 1. Conditioning vector
        t_emb = self.time_embed(timesteps)
        c = self.cond_mlp(torch.cat([s_t, t_emb], dim=-1))

        # 2. Token embedding + Positional embedding
        x = self.token_embed(z_k) + self.pos_embed

        # 3. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Final projection
        hat_x0 = self.final_layer(x, c)
        return hat_x0

    def forward(
        self,
        path_future: torch.Tensor,
        s_t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward training step: adds noise and computes total loss = MSE + lambda_fde * FDE^2.

        Args:
            path_future: Ground-truth trajectory (B, H, 4)
            s_t: Frozen World Model latent state (B, latent_dim)
            timesteps: Optional diffusion step indices (B,)
            noise: Optional Gaussian noise tensor (B, H, 4)

        Returns:
            Dictionary containing losses, pred_path, and metrics
        """
        batch_size = path_future.shape[0]
        device = path_future.device

        if timesteps is None:
            timesteps = torch.randint(
                0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long
            )

        if noise is None:
            noise = torch.randn_like(path_future)

        # Forward diffusion: q(z_k | z_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        z_k = sqrt_alpha * path_future + sqrt_one_minus_alpha * noise

        # Denoising network predicts denoised trajectory hat_path (x_0)
        hat_path = self.forward_dit(z_k, timesteps, s_t)

        # Loss 1: MSE reconstruction loss
        loss_mse = F.mse_loss(hat_path, path_future)

        # Loss 2: Differentiable FDE^2 penalty on the final step H (pose trajectories only;
        # an action-diffusion baseline with traj_dim=2 has no angle, so it uses MSE alone)
        if self.traj_dim >= 4:
            loss_fde_sq = compute_fde_torch(
                target_path=path_future,
                pred_path=hat_path,
                w_xy=self.w_xy,
                w_theta=self.w_theta,
                return_squared=True,
                reduction="mean",
            )
        else:
            loss_fde_sq = torch.zeros((), device=device)

        total_loss = loss_mse + self.lambda_fde * loss_fde_sq
        fde_value = torch.sqrt(loss_fde_sq.detach() + 1e-8)

        return {
            "total_loss": total_loss,
            "loss_mse": loss_mse,
            "loss_fde": loss_fde_sq,
            "fde_eval": fde_value,
            "pred_path": hat_path,
            "noisy_path": z_k,
        }

    @torch.no_grad()
    def sample(
        self,
        s_t: torch.Tensor,
        num_inference_steps: int = 10,
        eta: float = 0.0,
        current_pose_4d: Optional[torch.Tensor] = None,
        best_of_n: int = 1,
        goal_4d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast DDIM deterministic sampling with optional Inpainting and Best-of-N candidate ranking.

        Args:
            s_t: Latent state from frozen World Model (B, latent_dim)
            num_inference_steps: Number of denoising steps (default 10)
            eta: DDIM stochasticity factor (0.0 = deterministic)
            current_pose_4d: Optional true block pose [x, y, sin, cos] in [-1, 1]^4 to lock step 0
            best_of_n: Number of candidate trajectories to sample in parallel and rank (default 1)
            goal_4d: Target goal pose in [-1, 1]^4 (default [0, 0, sin(pi/4), cos(pi/4)])

        Returns:
            Denoised trajectory hat_path of shape (B, H, 4) in [-1, 1]^4
        """
        device = s_t.device
        orig_batch_size = s_t.shape[0]

        # Best-of-N expansion for single-state planning
        if best_of_n > 1 and orig_batch_size == 1:
            s_t = s_t.repeat(best_of_n, 1)
            batch_size = best_of_n
            if current_pose_4d is not None and current_pose_4d.shape[0] == 1:
                current_pose_4d = current_pose_4d.repeat(best_of_n, 1)
        else:
            batch_size = orig_batch_size

        # Initial standard Gaussian noise z_K
        z = torch.randn((batch_size, self.horizon, self.traj_dim), device=device)

        # Time sub-sequence for DDIM
        step_stride = self.diffusion_steps // num_inference_steps
        time_steps = list(range(0, self.diffusion_steps, step_stride))[::-1]

        for i, t in enumerate(time_steps):
            # Inpainting constraint: anchor step 0 to the physical current pose
            if current_pose_4d is not None:
                sqrt_a = self.sqrt_alphas_cumprod[t]
                sqrt_one_minus_a = self.sqrt_one_minus_alphas_cumprod[t]
                noise_0 = torch.randn_like(current_pose_4d)
                z[:, 0] = sqrt_a * current_pose_4d + sqrt_one_minus_a * noise_0

            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            hat_x0 = self.forward_dit(z, t_tensor, s_t)

            # Inpainting constraint on predicted x0 step 0
            if current_pose_4d is not None:
                hat_x0[:, 0] = current_pose_4d

            alpha_t = self.alphas_cumprod[t]
            t_prev = time_steps[i + 1] if i + 1 < len(time_steps) else -1
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Reconstruct eps from hat_x0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            eps = (z - sqrt_alpha_t * hat_x0) / (sqrt_one_minus_alpha_t + 1e-8)

            # DDIM update step
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * eps
            z_prev = torch.sqrt(alpha_prev) * hat_x0 + dir_xt
            if eta > 0.0 and t_prev >= 0:
                z_prev = z_prev + sigma * torch.randn_like(z)

            z = z_prev

        # Clamp output to valid [-1, 1] bounds
        z = torch.clamp(z, -1.0, 1.0)

        # Enforce exact initial pose at step 0
        if current_pose_4d is not None:
            z[:, 0] = current_pose_4d

        # Best-of-N Candidate Selection (pose trajectories only)
        if best_of_n > 1 and orig_batch_size == 1 and self.traj_dim >= 4:
            if goal_4d is None:
                sin_g = math.sin(math.pi / 4)
                cos_g = math.cos(math.pi / 4)
                goal_4d = torch.tensor([0.0, 0.0, sin_g, cos_g], device=device)

            # 1. Terminal distance error to goal (x, y)
            term_dist = (z[:, -1, :2] - goal_4d[:2]).pow(2).sum(dim=-1)

            # 2. Terminal orientation cosine error
            ang_cos = z[:, -1, 2] * goal_4d[2] + z[:, -1, 3] * goal_4d[3]
            ang_err = 1.0 - torch.clamp(ang_cos, -1.0, 1.0)

            # 3. Path smoothness (sum of consecutive displacement jumps)
            smooth_cost = (z[:, 1:, :2] - z[:, :-1, :2]).pow(2).sum(dim=(-1, -2))

            # Composite ranking score (lower is better)
            total_scores = 3.0 * term_dist + 2.0 * ang_err + 0.05 * smooth_cost
            best_idx = torch.argmin(total_scores).item()
            z = z[best_idx : best_idx + 1]

        return z[:orig_batch_size]

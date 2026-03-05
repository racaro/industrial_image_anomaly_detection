"""
DiffusionModel: DDPM for anomaly detection (reconstruction-based).

Architecture:
    - U-Net backbone with timestep embedding, skip connections, and BatchNorm
    - 4-level encoder: 256→128→64→32→16 spatial dims
    - Symmetric decoder with additive skip connections
    - Sinusoidal position embedding for timestep conditioning

Anomaly detection approach:
    - Train on normal images only (unsupervised)
    - At test time, add noise to input and denoise via reverse process
    - Normal images are reconstructed well; anomalies yield higher error
    - Anomaly score = MSE(original, reconstruction)

Input:  (B, 3, 256, 256)
Output: (B, 3, 256, 256) reconstructed image

References:
    - Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
    - Wyatt et al. "AnoDDPM: Anomaly Detection with DDPM" (CVPR-W 2022)
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position/timestep embedding (Vaswani et al., 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps.

        Returns:
            (B, dim) sinusoidal embedding.
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm + ReLU and optional time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # Add timestep embedding (broadcast over spatial dims)
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        return self.conv2(h)


class DownBlock(nn.Module):
    """Encoder block: ConvBlock + 2× downsampling."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x, t_emb)
        return self.downsample(h), h  # downsampled, skip


class UpBlock(nn.Module):
    """Decoder block: 2× upsample + ConvBlock with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        return self.conv(h, t_emb)


class UNet(nn.Module):
    """
    U-Net with timestep conditioning for DDPM.

    Encoder: 3→64→128→256→256 with 4 downsampling stages (256→16).
    Bottleneck: 256 channels at 16×16.
    Decoder: mirrors encoder with additive skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        time_emb_dim = base_channels * 4

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc1 = DownBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder (in_ch, skip_ch, out_ch)
        self.dec4 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 4, time_emb_dim)
        self.dec3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)
        self.dec2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)
        self.dec1 = UpBlock(base_channels, base_channels, base_channels, time_emb_dim)

        # Output projection
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 256) noisy image.
            t: (B,) integer timesteps.

        Returns:
            (B, 3, 256, 256) predicted noise (ε-prediction formulation).
        """
        t_emb = self.time_mlp(t)

        # Encoder
        d1, s1 = self.enc1(x, t_emb)  # 128×128
        d2, s2 = self.enc2(d1, t_emb)  # 64×64
        d3, s3 = self.enc3(d2, t_emb)  # 32×32
        d4, s4 = self.enc4(d3, t_emb)  # 16×16

        # Bottleneck
        b = self.bottleneck(d4, t_emb)  # 16×16

        # Decoder
        u4 = self.dec4(b, s4, t_emb)  # 32×32
        u3 = self.dec3(u4, s3, t_emb)  # 64×64
        u2 = self.dec2(u3, s2, t_emb)  # 128×128
        u1 = self.dec1(u2, s1, t_emb)  # 256×256

        return self.out_conv(u1)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule from β_start to β_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021). Smoother than linear."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionModel(nn.Module):
    """
    DDPM-based anomaly detector with ε-prediction.

    Training:
        1. Sample clean image x_0.
        2. Sample random timestep t and noise ε.
        3. Create noisy image x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.
        4. Predict noise: ε̂ = UNet(x_t, t).
        5. Loss = MSE(ε, ε̂).

    Inference (reconstruction for anomaly detection):
        1. Add noise to input x_0 up to timestep t_start (partial noising).
        2. Run reverse process from t_start→0 to reconstruct.
        3. Normal images are well-reconstructed; anomalies are not.

    Args:
        timesteps: Number of diffusion timesteps (default: 1000).
        schedule: Noise schedule type — ``"linear"`` or ``"cosine"``.
        inference_steps: Number of steps for reverse process during inference.
        noise_level: Fraction of total timesteps used for noising at inference
                     (e.g., 0.3 means noise up to step 300/1000).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        inference_steps: int = 50,
        noise_level: float = 0.3,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.inference_steps = inference_steps
        self.noise_level = noise_level
        self.unet = UNet()

        # Precompute noise schedule buffers
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register as buffers (moved to device with model, not trained)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance for reverse process
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to x_0 at timestep t.

        Args:
            x_0: (B, C, H, W) clean images.
            t: (B,) timesteps.
            noise: Optional pre-generated noise.

        Returns:
            (noisy_image, noise) tuple.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε̂ from noisy image x_t at timestep t."""
        return self.unet(x_t, t)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single reverse diffusion step: denoise x_t → x_{t-1}.

        Args:
            x_t: (B, C, H, W) current noisy image.
            t: Current integer timestep.

        Returns:
            (B, C, H, W) slightly less noisy image.
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        eps_pred = self.predict_noise(x_t, t_tensor)

        # Compute x_{t-1} mean
        beta_t = self.betas[t]

        mean = self.sqrt_recip_alphas[t] * (x_t - (beta_t / self.sqrt_one_minus_alphas_cumprod[t]) * eps_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def reconstruct(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images via partial noising + reverse diffusion.

        Strategy (AnoDDPM-inspired):
            1. Add noise to x_0 up to timestep t_start (partial corruption).
            2. Iteratively denoise from t_start → 0.
            3. The model reconstructs normal structure; anomalies are lost.

        Args:
            x_0: (B, C, H, W) input images in [0, 1].

        Returns:
            (B, C, H, W) reconstructed images in [0, 1].
        """
        t_start = int(self.timesteps * self.noise_level)
        t_start = max(t_start, 1)

        # Forward: add noise up to t_start
        t_tensor = torch.full((x_0.shape[0],), t_start - 1, device=x_0.device, dtype=torch.long)
        x_t, _ = self.q_sample(x_0, t_tensor)

        # Determine which steps to denoise through
        step_size = max(1, t_start // self.inference_steps)
        timesteps = list(range(t_start - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # Reverse process
        for t in timesteps:
            x_t = self.p_sample(x_t, t)

        return x_t.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Unified forward pass.

        - **Training mode** (``t`` provided): predict noise ε̂ from (x_t, t).
        - **Eval mode**  (``t is None``): reconstruct x via reverse diffusion.
          This makes the model compatible with the shared evaluation pipeline.

        Args:
            x: (B, 3, 256, 256) input tensor.
            t: (B,) timesteps. If *None*, runs reconstruction.

        Returns:
            (B, 3, 256, 256) predicted noise (training) or reconstruction (eval).
        """
        if t is not None:
            return self.predict_noise(x, t)
        return self.reconstruct(x)

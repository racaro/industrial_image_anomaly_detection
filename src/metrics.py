"""
Evaluation metrics: SSIM computation and shared metric utilities.
"""

import torch
import torch.nn as nn


def compute_ssim_batch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Simplified SSIM per image (average over channels and pixels).
    x, y: (B, C, H, W) in [0, 1].
    Returns tensor (B,) with SSIM per image.
    """
    c1 = 0.01**2
    c2 = 0.03**2

    pad = window_size // 2
    pool = nn.AvgPool2d(window_size, stride=1, padding=pad)

    mu_x = pool(x)
    mu_y = pool(y)
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x2 = pool(x**2) - mu_x2
    sigma_y2 = pool(y**2) - mu_y2
    sigma_xy = pool(x * y) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = num / den
    return ssim_map.mean(dim=[1, 2, 3])

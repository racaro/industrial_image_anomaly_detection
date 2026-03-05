"""
Evaluation metrics: SSIM computation and shared metric utilities.
"""

import numpy as np
import torch
import torch.nn as nn


def _min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1].

    Args:
        x: 1-D array of raw scores.

    Returns:
        Normalized array (same shape). Returns zeros if range is 0.
    """
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-10) if rng > 0 else np.zeros_like(x)


def compute_combined_score(
    scores_mse: np.ndarray,
    scores_ssim: np.ndarray,
    scores_perceptual: np.ndarray,
    *,
    w_mse: float = 0.3,
    w_ssim: float = 0.3,
    w_perceptual: float = 0.4,
) -> np.ndarray:
    """Weighted combination of min-max-normalized anomaly scores.

    Default weights: ``0.3 × MSE + 0.3 × SSIM + 0.4 × Perceptual``.

    Args:
        scores_mse: Raw MSE anomaly scores (B,).
        scores_ssim: Raw SSIM anomaly scores (B,) — **already inverted**
            (i.e. 1 − SSIM so higher = more anomalous).
        scores_perceptual: Raw perceptual anomaly scores (B,).
        w_mse: Weight for MSE component.
        w_ssim: Weight for SSIM component.
        w_perceptual: Weight for perceptual component.

    Returns:
        Combined score array (B,).
    """
    return (
        w_mse * _min_max_normalize(scores_mse)
        + w_ssim * _min_max_normalize(scores_ssim)
        + w_perceptual * _min_max_normalize(scores_perceptual)
    )


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

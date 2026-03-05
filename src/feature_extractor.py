"""
Pre-trained VGG-16 Feature Extractor for Perceptual Anomaly Scoring.

Extracts intermediate features from a frozen VGG-16 network.
Used for:
  1. **Perceptual anomaly scoring** – compare features of original vs reconstruction
  2. **Perceptual loss** during training (optional)

Feature layers (after ReLU):
  - relu1_2 (index 3): 64 × H/2  × W/2   – low-level edges/textures
  - relu2_2 (index 8): 128 × H/4  × W/4   – mid-level textures
  - relu3_3 (index 15): 256 × H/8  × W/8  – higher-level patterns
  - relu4_3 (index 22): 512 × H/16 × W/16 – semantic features
"""

import torch
import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG-16 multi-layer feature extractor.

    Extracts features at configurable intermediate layers.
    Input images are normalised from [0, 1] to ImageNet statistics internally.
    """

    # Default VGG-16 layer indices (after ReLU)
    DEFAULT_LAYERS = (3, 8, 15, 22)
    # Weights for combining per-layer distances
    DEFAULT_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    def __init__(self, layer_indices=None):
        super().__init__()
        layer_indices = layer_indices or self.DEFAULT_LAYERS

        # Load pre-trained VGG-16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Build sequential slices for each target layer
        features = list(vgg.features.children())
        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in layer_indices:
            self.slices.append(nn.Sequential(*features[prev : layer_idx + 1]))
            prev = layer_idx + 1

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: (B, 3, H, W) tensor in [0, 1].

        Returns:
            List of feature tensors, one per target layer.
        """
        x = (x - self.mean) / self.std
        features = []
        for s in self.slices:
            x = s(x)
            features.append(x)
        return features


def compute_perceptual_score(
    extractor: VGGFeatureExtractor,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """
    Compute per-image perceptual distance between original and reconstruction.

    Uses L2 distance in VGG feature space, summed across selected layers.

    Args:
        extractor: Frozen VGG feature extractor.
        original:  (B, 3, H, W) in [0, 1].
        reconstruction: (B, 3, H, W) in [0, 1].
        weights: Per-layer weighting (default: equal weights).

    Returns:
        Tensor (B,) with perceptual anomaly score per image (higher = more anomalous).
    """
    if weights is None:
        weights = VGGFeatureExtractor.DEFAULT_WEIGHTS

    feats_orig = extractor(original)
    feats_recon = extractor(reconstruction)

    score = torch.zeros(original.size(0), device=original.device)
    for w, fo, fr in zip(weights, feats_orig, feats_recon, strict=False):
        diff = (fo - fr) ** 2
        score += w * diff.mean(dim=[1, 2, 3])

    return score

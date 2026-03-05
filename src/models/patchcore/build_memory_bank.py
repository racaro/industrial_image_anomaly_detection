"""
PatchCore anomaly detection with pre-trained features.

Implements the PatchCore algorithm (Roth et al., 2022) using a pre-trained
WideResNet-50 backbone. Each category gets its own memory bank of normal
patch features, and anomaly scoring uses nearest-neighbor distance.

Key design:
    1. Extract multi-scale patch features from WideResNet-50 (layers 2 & 3)
    2. Build a memory bank per category from train/good images
    3. Subsample the memory bank with coreset selection for efficiency
    4. Score test images by nearest-neighbor distance in feature space
"""

import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_WORKERS,
    OUTPUTS_DIR,
    ensure_dataset,
    set_seed,
)
from src.dataset import (
    AnomalyImageDataset,
    collect_image_paths,
    get_categories,
    validate_images,
)
from src.logger import get_logger

logger = get_logger(__name__)

PATCHCORE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "patchcore")
"""Root directory for PatchCore memory banks."""

CORESET_RATIO = 0.1
"""Fraction of patch features to keep via coreset subsampling (0.1 = 10%)."""

BACKBONE_NAME = "wide_resnet50_2"
"""Pre-trained backbone model name."""


class PatchCoreFeatureExtractor(nn.Module):
    """
    Multi-scale patch feature extractor using WideResNet-50.

    Extracts intermediate features from layers 2 and 3, then concatenates
    them at a common spatial resolution to form rich patch descriptors.

    Feature dimensions:
        - Layer 2: 512 channels (stride 8)
        - Layer 3: 1024 channels (stride 16) → upsampled to stride 8
        - Concatenated: 1536 channels per spatial position

    All parameters are frozen (no training required).
    """

    def __init__(self) -> None:
        super().__init__()

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)

        # Build sequential feature slices
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4, 256 ch
        self.layer2 = backbone.layer2  # stride 8, 512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract concatenated multi-scale patch features.

        Args:
            x: (B, 3, H, W) tensor in [0, 1].

        Returns:
            (B, 1536, H/8, W/8) tensor of patch features.
        """
        x = (x - self.mean) / self.std

        x = self.layer0(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)  # (B, 512, H/8, W/8)
        feat3 = self.layer3(feat2)  # (B, 1024, H/16, W/16)

        # Upsample layer3 to match layer2 spatial resolution
        feat3_up = F.interpolate(
            feat3,
            size=feat2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate → 1536 channels per patch
        features = torch.cat([feat2, feat3_up], dim=1)
        return features


def coreset_subsampling(
    features: np.ndarray,
    ratio: float = CORESET_RATIO,
    max_samples: int = 5000,
) -> np.ndarray:
    """
    Two-stage coreset subsampling to reduce memory bank size efficiently.

    Stage 1: Random pre-selection to reduce N to a manageable pool
             (max PRE_POOL_SIZE features).
    Stage 2: Greedy farthest-point selection on the reduced pool using
             GPU-accelerated distance computation when available.

    This avoids the O(N × target) bottleneck on very large feature sets
    (e.g., 200K+ patches) while preserving representative coverage.

    Args:
        features: (N, D) array of feature vectors.
        ratio: Fraction of features to keep.
        max_samples: Hard cap on coreset size.

    Returns:
        (M, D) subsampled feature array where M ≈ N × ratio.
    """
    PRE_POOL_SIZE = 20000  # Random pre-selection cap

    n = features.shape[0]
    target = min(int(n * ratio), max_samples)

    if target >= n:
        return features

    logger.info(
        "Coreset subsampling: %d → %d features (%.1f%%)",
        n,
        target,
        100 * target / n,
    )

    rng = np.random.default_rng(42)

    # Stage 1: Random pre-selection if N is too large for greedy
    if n > PRE_POOL_SIZE:
        logger.info("  Stage 1: Random pre-selection %d → %d", n, PRE_POOL_SIZE)
        indices = rng.choice(n, size=PRE_POOL_SIZE, replace=False)
        features = features[indices]
        n = PRE_POOL_SIZE

    # Stage 2: Greedy farthest-point selection (GPU-accelerated)
    logger.info("  Stage 2: Greedy farthest-point selection → %d", target)

    # Use GPU if available for faster distance computation
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        feat_tensor = torch.from_numpy(features).float().cuda()
    else:
        feat_tensor = torch.from_numpy(features).float()

    selected_indices: list[int] = [int(rng.integers(n))]
    min_distances = torch.full((n,), float("inf"), device=feat_tensor.device)

    for _ in tqdm(range(target - 1), desc="    Coreset", leave=False):
        last = feat_tensor[selected_indices[-1]].unsqueeze(0)  # (1, D)
        dists = torch.norm(feat_tensor - last, dim=1)  # (N,)
        min_distances = torch.minimum(min_distances, dists)
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)

    # Gather selected features back to numpy
    selected_features = feat_tensor[selected_indices].cpu().numpy()
    return selected_features


def build_memory_bank(
    extractor: PatchCoreFeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
    coreset_ratio: float = CORESET_RATIO,
) -> np.ndarray:
    """
    Build a memory bank of normal patch features for one category.

    Args:
        extractor: Frozen WideResNet-50 feature extractor.
        dataloader: DataLoader with train/good images.
        device: Torch device.
        coreset_ratio: Fraction of features to keep.

    Returns:
        (M, 1536) array of representative normal patch features.
    """
    all_features: list[np.ndarray] = []

    for imgs, _ in tqdm(dataloader, desc="    Extracting features", leave=False):
        imgs = imgs.to(device)
        features = extractor(imgs)  # (B, 1536, H', W')

        # Reshape to (B * H' * W', 1536)
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        all_features.append(features.cpu().numpy())

    all_features_np = np.concatenate(all_features, axis=0)
    logger.info(
        "Raw patch features: %d × %d",
        all_features_np.shape[0],
        all_features_np.shape[1],
    )

    # Coreset subsampling
    memory_bank = coreset_subsampling(all_features_np, ratio=coreset_ratio)
    logger.info(
        "Memory bank size: %d × %d",
        memory_bank.shape[0],
        memory_bank.shape[1],
    )

    return memory_bank


def main(
    selected_categories: list[str] | None = None,
    coreset_ratio: float = CORESET_RATIO,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Build PatchCore memory banks for all (or selected) categories.

    Args:
        selected_categories: Specific categories. None = all.
        coreset_ratio: Fraction of features to keep.
        batch_size: Batch size for feature extraction.

    Returns:
        List of summary dicts (one per category).
    """
    set_seed()
    ensure_dataset()

    all_categories = get_categories(DATASET_PATH)
    logger.info("Dataset categories found: %d", len(all_categories))

    if selected_categories:
        invalid = [c for c in selected_categories if c not in all_categories]
        if invalid:
            raise ValueError(f"Unknown categories: {invalid}.")
        categories = selected_categories
    else:
        categories = all_categories

    logger.info(
        "Building PatchCore memory banks for %d categories on %s",
        len(categories),
        DEVICE,
    )

    # Load backbone
    logger.info("Loading WideResNet-50 backbone...")
    extractor = PatchCoreFeatureExtractor().to(DEVICE)
    extractor.eval()
    logger.info("Backbone ready (%.1fM params, frozen).", sum(p.numel() for p in extractor.parameters()) / 1e6)

    use_pin_memory = torch.cuda.is_available()
    summaries: list[dict] = []
    total_start = time.time()

    for i, category in enumerate(categories, 1):
        logger.info("=" * 60)
        logger.info("  [%d/%d] Building memory bank: %s", i, len(categories), category.upper())
        logger.info("=" * 60)

        # Collect training images
        image_data = collect_image_paths(DATASET_PATH, [category], subfolder=os.path.join("train", "good"))
        if not image_data:
            logger.warning("[%s] No training images — skipping.", category)
            continue

        image_data, _ = validate_images(image_data)
        logger.info("[%s] Valid training images: %d", category, len(image_data))

        # DataLoader
        dataset = AnomalyImageDataset(image_data, IMG_HEIGHT, IMG_WIDTH)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=use_pin_memory,
        )

        # Build memory bank
        start = time.time()
        memory_bank = build_memory_bank(extractor, dataloader, DEVICE, coreset_ratio)
        elapsed = time.time() - start

        # Save
        save_dir = os.path.join(PATCHCORE_OUTPUT_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        bank_path = os.path.join(save_dir, "memory_bank.pt")
        torch.save(torch.from_numpy(memory_bank), bank_path)
        logger.info("[%s] Memory bank saved → %s", category, bank_path)

        summary = {
            "category": category,
            "num_images": len(image_data),
            "raw_patches": int(memory_bank.shape[0] / coreset_ratio) if coreset_ratio < 1 else memory_bank.shape[0],
            "memory_bank_size": memory_bank.shape[0],
            "feature_dim": memory_bank.shape[1],
            "coreset_ratio": coreset_ratio,
            "elapsed_seconds": round(elapsed, 1),
        }
        summaries.append(summary)

        logger.info(
            "[%s] Done — %d patches → %d coreset features (%.0fs)",
            category,
            summary["raw_patches"],
            summary["memory_bank_size"],
            elapsed,
        )

    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info("  PATCHCORE MEMORY BANK BUILDING COMPLETE")
    logger.info("=" * 60)
    logger.info("Categories processed: %d / %d", len(summaries), len(categories))
    logger.info("Total time: %.1f min", total_elapsed / 60)

    for s in summaries:
        logger.info(
            "  %-15s  images: %4d  bank: %5d × %d  (%.0fs)",
            s["category"],
            s["num_images"],
            s["memory_bank_size"],
            s["feature_dim"],
            s["elapsed_seconds"],
        )

    # Save global summary
    global_summary = {
        "backbone": BACKBONE_NAME,
        "coreset_ratio": coreset_ratio,
        "feature_dim": 1536,
        "device": str(DEVICE),
        "total_time_seconds": round(total_elapsed, 1),
        "categories": summaries,
    }
    summary_path = os.path.join(PATCHCORE_OUTPUT_DIR, "build_summary.json")
    os.makedirs(PATCHCORE_OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved → %s", summary_path)

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PatchCore memory banks from pre-trained WideResNet-50 features."
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=None,
        help="Specific categories. Default: all.",
    )
    parser.add_argument(
        "--coreset-ratio",
        "-cr",
        type=float,
        default=CORESET_RATIO,
        help=f"Coreset subsampling ratio (default: {CORESET_RATIO}).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    main(
        selected_categories=args.categories,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
    )

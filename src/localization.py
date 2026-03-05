"""
Anomaly localization: pixel-level heatmaps from PatchCore.

Generates anomaly maps by computing per-patch nearest-neighbor distances
to the memory bank, then upsampling and smoothing to produce pixel-aligned
heatmaps. Supports overlay visualization on original images.

Architecture:
    1. Extract patch features from WideResNet-50 (1536-dim at stride 8)
    2. Compute per-patch distance to k-nearest neighbors in memory bank
    3. Reshape distances to spatial map (H/8 × W/8)
    4. Upsample to original resolution (H × W)
    5. Apply Gaussian smoothing for visual coherence
    6. Normalize to [0, 1] for consistent visualization
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from src.config import (
    DATASET_PATH,
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    set_seed,
)
from src.dataset import EvalImageDataset, collect_test_images
from src.logger import get_logger
from src.models.patchcore.build_memory_bank import (
    PATCHCORE_OUTPUT_DIR,
    PatchCoreFeatureExtractor,
)

logger = get_logger(__name__)

LOCALIZATION_DIR = os.path.join(PATCHCORE_OUTPUT_DIR, "localization")
"""Root directory for localization outputs."""

GAUSSIAN_SIGMA = 4.0
"""Sigma for Gaussian smoothing of anomaly maps (in pixels at output resolution)."""

TOP_K_SCORING = 3
"""Number of nearest neighbors for patch-level scoring."""

MAX_IMAGES_PER_CATEGORY = 10
"""Maximum images to generate localization visualizations for per category."""

COLORMAP = "inferno"
"""Matplotlib colormap for heatmaps. 'inferno' emphasizes anomalous regions clearly."""

OVERLAY_ALPHA = 0.45
"""Transparency of the heatmap overlay on the original image."""


def compute_anomaly_map(
    extractor: PatchCoreFeatureExtractor,
    image: torch.Tensor,
    memory_bank: torch.Tensor,
    top_k: int = TOP_K_SCORING,
    sigma: float = GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, float]:
    """
    Compute a pixel-level anomaly heatmap for a single image.

    For each spatial position (patch) in the feature map, computes the
    distance to the k-nearest neighbors in the memory bank. The resulting
    distance map is upsampled to the original resolution and smoothed
    with a Gaussian filter.

    Args:
        extractor: Frozen WideResNet-50 feature extractor (on device).
        image: (1, 3, H, W) tensor in [0, 1] (on device).
        memory_bank: (M, D) tensor of normal features (on device).
        top_k: Number of nearest neighbors for scoring.
        sigma: Gaussian smoothing sigma.

    Returns:
        Tuple of:
            anomaly_map: (H, W) normalized anomaly map in [0, 1].
            image_score: Scalar anomaly score (max of raw patch distances).
    """
    with torch.no_grad():
        features = extractor(image)  # (1, 1536, H', W')
        _, c, h_feat, w_feat = features.shape

        # (1, H'*W', 1536)
        patches = features.permute(0, 2, 3, 1).reshape(1, h_feat * w_feat, c)

        # (H'*W', M) — distance to each memory bank entry
        dists = torch.cdist(
            patches.squeeze(0).unsqueeze(0),
            memory_bank.unsqueeze(0),
        ).squeeze(0)  # (H'*W', M)

        # Per-patch score: mean of top-k nearest neighbor distances
        if top_k == 1:
            patch_scores, _ = dists.min(dim=1)  # (H'*W',)
        else:
            topk_dists, _ = dists.topk(top_k, dim=1, largest=False)
            patch_scores = topk_dists.mean(dim=1)  # (H'*W',)

        # Image-level score
        image_score = float(patch_scores.max().item())

        # Reshape to spatial map
        score_map = patch_scores.reshape(1, 1, h_feat, w_feat)  # (1, 1, H', W')

        # Upsample to original resolution
        score_map = (
            F.interpolate(
                score_map,
                size=(IMG_HEIGHT, IMG_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )  # (H, W)

    # Gaussian smoothing
    score_map = gaussian_filter(score_map, sigma=sigma)

    # Normalize to [0, 1]
    s_min, s_max = score_map.min(), score_map.max()
    if s_max - s_min > 1e-8:
        score_map = (score_map - s_min) / (s_max - s_min)
    else:
        score_map = np.zeros_like(score_map)

    return score_map, image_score


def compute_anomaly_maps_batch(
    extractor: PatchCoreFeatureExtractor,
    images: torch.Tensor,
    memory_bank: torch.Tensor,
    top_k: int = TOP_K_SCORING,
    sigma: float = GAUSSIAN_SIGMA,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Compute anomaly maps for a batch of images.

    Args:
        extractor: Frozen WideResNet-50 feature extractor.
        images: (B, 3, H, W) tensor in [0, 1].
        memory_bank: (M, D) tensor of normal features.
        top_k: Number of nearest neighbors.
        sigma: Gaussian smoothing sigma.

    Returns:
        Tuple of:
            maps: List of B (H, W) anomaly maps in [0, 1].
            scores: (B,) array of image-level scores.
    """
    maps: list[np.ndarray] = []
    scores: list[float] = []

    for i in range(images.shape[0]):
        amap, score = compute_anomaly_map(extractor, images[i : i + 1], memory_bank, top_k, sigma)
        maps.append(amap)
        scores.append(score)

    return maps, np.array(scores)


def render_heatmap(anomaly_map: np.ndarray, cmap: str = COLORMAP) -> np.ndarray:
    """
    Convert a single-channel anomaly map to an RGB heatmap image.

    Args:
        anomaly_map: (H, W) array in [0, 1].
        cmap: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    colormap = plt.get_cmap(cmap)
    heatmap_rgba = colormap(anomaly_map)  # (H, W, 4) float [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    return heatmap_rgb


def render_overlay(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    cmap: str = COLORMAP,
) -> np.ndarray:
    """
    Overlay anomaly heatmap on the original image.

    Args:
        original: (H, W, 3) uint8 RGB image.
        anomaly_map: (H, W) array in [0, 1].
        alpha: Heatmap opacity.
        cmap: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 RGB overlay.
    """
    heatmap = render_heatmap(anomaly_map, cmap).astype(np.float32)
    base = original.astype(np.float32)
    overlay = (1 - alpha) * base + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_localization_grid(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    image_score: float,
    label_name: str,
    save_path: str,
    cmap: str = COLORMAP,
    alpha: float = OVERLAY_ALPHA,
) -> None:
    """
    Save a 3-panel visualization: Original | Heatmap | Overlay.

    Args:
        original: (H, W, 3) uint8 RGB image.
        anomaly_map: (H, W) normalized anomaly map.
        image_score: Scalar anomaly score.
        label_name: 'good' or 'anomaly'.
        save_path: Output file path.
        cmap: Colormap for heatmap.
        alpha: Overlay transparency.
    """
    render_heatmap(anomaly_map, cmap)
    overlay = render_overlay(original, anomaly_map, alpha, cmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Anomaly Map", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    label_color = "#e74c3c" if label_name == "anomaly" else "#2ecc71"
    fig.suptitle(
        f"Label: {label_name.upper()}  |  Score: {image_score:.2f}",
        fontsize=14,
        fontweight="bold",
        color=label_color,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_category_summary(
    grids: list[dict],
    category: str,
    save_dir: str,
) -> None:
    """
    Save a summary figure with top-N most anomalous and top-N most normal images.

    Args:
        grids: List of dicts with 'anomaly_map', 'original', 'score', 'label', 'filename'.
        category: Category name.
        save_dir: Output directory.
    """
    if not grids:
        return

    # Sort by score
    sorted_grids = sorted(grids, key=lambda g: g["score"], reverse=True)

    # Top anomalous (up to 5)
    top_anomalous = sorted_grids[:5]
    # Most normal (up to 5)
    most_normal = sorted(grids, key=lambda g: g["score"])[:5]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    for col, data in enumerate(top_anomalous):
        overlay = render_overlay(data["original"], data["anomaly_map"])
        axes[0, col].imshow(overlay)
        label_color = "#e74c3c" if data["label"] == "anomaly" else "#2ecc71"
        axes[0, col].set_title(
            f"{data['label'].upper()}\nScore: {data['score']:.1f}",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )
        axes[0, col].axis("off")

    for col in range(len(top_anomalous), 5):
        axes[0, col].axis("off")

    for col, data in enumerate(most_normal):
        overlay = render_overlay(data["original"], data["anomaly_map"])
        axes[1, col].imshow(overlay)
        label_color = "#e74c3c" if data["label"] == "anomaly" else "#2ecc71"
        axes[1, col].set_title(
            f"{data['label'].upper()}\nScore: {data['score']:.1f}",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )
        axes[1, col].axis("off")

    for col in range(len(most_normal), 5):
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Most Anomalous", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Most Normal", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Anomaly Localization Summary — {category.upper()}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "summary.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_localization(
    selected_categories: list[str] | None = None,
    only_anomalies: bool = False,
    max_per_category: int = MAX_IMAGES_PER_CATEGORY,
    top_k: int = TOP_K_SCORING,
    sigma: float = GAUSSIAN_SIGMA,
) -> dict:
    """
    Run anomaly localization on test images for all (or selected) categories.

    For each category:
        1. Loads the pre-built memory bank
        2. Scores test images and generates anomaly maps
        3. Saves individual grid visualizations up to max_per_category
        4. Generates a summary figure with most/least anomalous images

    Args:
        selected_categories: Subset of categories to process. None = all.
        only_anomalies: If True, only generate heatmaps for anomaly images.
        max_per_category: Max individual visualizations per category.
        top_k: k-NN parameter for scoring.
        sigma: Gaussian smoothing sigma.

    Returns:
        Summary dict with per-category statistics.
    """
    set_seed()

    # Collect test images
    all_records = collect_test_images(DATASET_PATH)
    records_by_cat: dict[str, list[dict]] = {}
    for r in all_records:
        records_by_cat.setdefault(r["category"], []).append(r)

    categories = sorted(records_by_cat.keys())
    if selected_categories:
        categories = [c for c in categories if c in selected_categories]
        invalid = [c for c in selected_categories if c not in records_by_cat]
        if invalid:
            logger.warning("Unknown categories (skipped): %s", invalid)

    logger.info(
        "Anomaly localization for %d categories on %s",
        len(categories),
        DEVICE,
    )

    # Load backbone
    logger.info("Loading WideResNet-50 backbone...")
    extractor = PatchCoreFeatureExtractor().to(DEVICE)
    extractor.eval()
    logger.info("Backbone ready.")

    summary: dict[str, dict] = {}

    for i, category in enumerate(categories, 1):
        logger.info("=" * 60)
        logger.info("  [%d/%d] Localizing: %s", i, len(categories), category.upper())
        logger.info("=" * 60)

        # Load memory bank
        bank_path = os.path.join(PATCHCORE_OUTPUT_DIR, category, "memory_bank.pt")
        if not os.path.isfile(bank_path):
            logger.warning("[%s] No memory bank — skipping.", category)
            continue

        memory_bank = torch.load(bank_path, map_location=DEVICE, weights_only=True)
        if not isinstance(memory_bank, torch.Tensor):
            memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
        memory_bank = memory_bank.to(DEVICE)

        cat_records = records_by_cat.get(category, [])
        if only_anomalies:
            cat_records = [r for r in cat_records if r["label"] == 1]

        if not cat_records:
            logger.warning("[%s] No test images.", category)
            continue

        logger.info(
            "[%s] Processing %d test images (bank: %d × %d)",
            category,
            len(cat_records),
            *memory_bank.shape,
        )

        # Prepare output directory
        cat_loc_dir = os.path.join(LOCALIZATION_DIR, category)
        os.makedirs(cat_loc_dir, exist_ok=True)

        # Process images
        dataset = EvalImageDataset(cat_records, IMG_HEIGHT, IMG_WIDTH)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        grids_data: list[dict] = []

        for _img_idx, (img_tensor, label_tensor, rec_idx) in enumerate(loader):
            img_tensor = img_tensor.to(DEVICE)
            label = int(label_tensor.item())
            record = cat_records[int(rec_idx.item())]
            filename = os.path.splitext(os.path.basename(record["path"]))[0]
            label_name = "anomaly" if label == 1 else "good"

            # Compute anomaly map
            anomaly_map, image_score = compute_anomaly_map(extractor, img_tensor, memory_bank, top_k, sigma)

            # Load original image for visualization
            original = np.array(
                Image.open(record["path"]).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            )

            grids_data.append(
                {
                    "anomaly_map": anomaly_map,
                    "original": original,
                    "score": image_score,
                    "label": label_name,
                    "filename": filename,
                }
            )

        # Save the most interesting individual grids:
        # top-N highest scoring + top-N lowest scoring (mix of anomalies + goods)
        half = max(max_per_category // 2, 1)
        sorted_by_score = sorted(grids_data, key=lambda g: g["score"], reverse=True)
        top_anomalous_grids = sorted_by_score[:half]
        most_normal_grids = sorted_by_score[-half:]
        grids_to_save = {id(g): g for g in top_anomalous_grids + most_normal_grids}

        saved_count = 0
        for g in grids_to_save.values():
            grid_path = os.path.join(cat_loc_dir, f"{g['label']}_{g['filename']}_grid.png")
            save_localization_grid(
                g["original"],
                g["anomaly_map"],
                g["score"],
                g["label"],
                grid_path,
            )
            saved_count += 1

        # Category summary figure
        save_category_summary(grids_data, category, cat_loc_dir)

        # Statistics
        scores_by_label: dict[str, list[float]] = {"good": [], "anomaly": []}
        for g in grids_data:
            scores_by_label[g["label"]].append(g["score"])

        cat_summary = {
            "category": category,
            "total_images": len(grids_data),
            "individual_grids_saved": saved_count,
        }
        for lbl in ("good", "anomaly"):
            if scores_by_label[lbl]:
                cat_summary[f"score_{lbl}_mean"] = round(float(np.mean(scores_by_label[lbl])), 2)
                cat_summary[f"score_{lbl}_std"] = round(float(np.std(scores_by_label[lbl])), 2)

        summary[category] = cat_summary
        logger.info(
            "[%s] Done — %d grids saved, summary generated",
            category,
            saved_count,
        )

    summary_path = os.path.join(LOCALIZATION_DIR, "localization_summary.json")
    os.makedirs(LOCALIZATION_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("  LOCALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Categories processed: %d", len(summary))
    logger.info("Results saved to: %s", LOCALIZATION_DIR)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anomaly localization heatmaps using PatchCore.")
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=None,
        help="Specific categories to process. Default: all.",
    )
    parser.add_argument(
        "--only-anomalies",
        action="store_true",
        help="Only generate heatmaps for anomaly test images.",
    )
    parser.add_argument(
        "--max-per-category",
        "-m",
        type=int,
        default=MAX_IMAGES_PER_CATEGORY,
        help=f"Max individual visualizations per category (default: {MAX_IMAGES_PER_CATEGORY}).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=TOP_K_SCORING,
        help=f"k-NN parameter (default: {TOP_K_SCORING}).",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=GAUSSIAN_SIGMA,
        help=f"Gaussian smoothing sigma (default: {GAUSSIAN_SIGMA}).",
    )
    args = parser.parse_args()

    run_localization(
        selected_categories=args.categories,
        only_anomalies=args.only_anomalies,
        max_per_category=args.max_per_category,
        top_k=args.top_k,
        sigma=args.sigma,
    )

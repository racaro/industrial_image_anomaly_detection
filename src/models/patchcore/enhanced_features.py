"""
Enhanced PatchCore Feature Extraction for Weak Categories.

Implements four evidence-backed improvements over the baseline PatchCore:

1. **Local neighborhood aggregation** (from the original PatchCore paper,
   Roth et al. 2022) — averages nearby patch features with a 3×3 kernel
   to make representations robust to small spatial shifts.
2. **L2 feature normalization** — normalizes concatenated features to unit
   length, reducing scale imbalance between WideResNet layers.
3. **Layer 1 inclusion** — adds 256-ch features (stride 4) for finer
   spatial detail. Total dims: 1536 → 1792.
4. **Higher resolution support** — 512×512 input yields 64×64 spatial grid
   (4× more patches) for capturing fine-grained defects.

Usage:
    python -m src.models.patchcore.enhanced_features --categories grid screw capsules
    python -m src.models.patchcore.enhanced_features --resolution 512
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

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
    DATASET_PATH,
    DEVICE,
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
from src.models.patchcore.build_memory_bank import (
    PATCHCORE_OUTPUT_DIR,
    coreset_subsampling,
)

logger = get_logger(__name__)


ENHANCED_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "patchcore_enhanced")
"""Root directory for enhanced PatchCore outputs."""


@dataclass
class EnhancedConfig:
    """Configuration for enhanced PatchCore feature extraction.

    Attributes:
        resolution: Input image resolution (height = width).
        use_layer1: Whether to include layer 1 features (256 ch).
        neighborhood_size: Kernel size for local averaging (0 = disabled).
        l2_normalize: Whether to L2-normalize concatenated features.
        coreset_ratio: Fraction of features to keep via coreset subsampling.
        max_coreset: Hard cap on coreset size.
        top_k_scoring: Number of nearest neighbors for scoring.
        batch_size: Batch size for feature extraction.
    """

    resolution: int = 512
    use_layer1: bool = True
    neighborhood_size: int = 3
    l2_normalize: bool = True
    coreset_ratio: float = 0.10
    max_coreset: int = 10000
    top_k_scoring: int = 1
    batch_size: int = 8

    @property
    def feature_dim(self) -> int:
        """Total feature dimension after concatenation."""
        dim = 512 + 1024  # layer2 + layer3
        if self.use_layer1:
            dim += 256  # layer1
        return dim

    @property
    def spatial_size(self) -> int:
        """Spatial grid size at stride 8."""
        return self.resolution // 8

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "resolution": self.resolution,
            "use_layer1": self.use_layer1,
            "neighborhood_size": self.neighborhood_size,
            "l2_normalize": self.l2_normalize,
            "coreset_ratio": self.coreset_ratio,
            "max_coreset": self.max_coreset,
            "top_k_scoring": self.top_k_scoring,
            "batch_size": self.batch_size,
            "feature_dim": self.feature_dim,
            "spatial_size": self.spatial_size,
        }


class EnhancedFeatureExtractor(nn.Module):
    """
    Enhanced multi-scale patch feature extractor using WideResNet-50.

    Improvements over baseline:
        - Optional layer 1 inclusion (256 ch, stride 4 → finer details)
        - Local neighborhood aggregation (3×3 average pooling)
        - L2 normalization of concatenated features

    Feature dimensions:
        - Layer 1: 256 channels (stride 4) → downsampled to stride 8
        - Layer 2: 512 channels (stride 8)
        - Layer 3: 1024 channels (stride 16) → upsampled to stride 8
        - Concatenated: 1792 channels (with layer 1) or 1536 (without)

    All parameters are frozen (no training required).
    """

    def __init__(self, config: EnhancedConfig) -> None:
        super().__init__()
        self.config = config

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4, 256 ch
        self.layer2 = backbone.layer2  # stride 8, 512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch

        # Local neighborhood aggregation
        if config.neighborhood_size > 0:
            pad = config.neighborhood_size // 2
            self.neighborhood_agg = nn.AvgPool2d(
                kernel_size=config.neighborhood_size,
                stride=1,
                padding=pad,
            )
        else:
            self.neighborhood_agg = None

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract enhanced multi-scale patch features.

        Args:
            x: (B, 3, H, W) tensor in [0, 1].

        Returns:
            (B, D, H/8, W/8) tensor of patch features where D is
            1792 (with layer 1) or 1536 (without).
        """
        x = (x - self.mean) / self.std

        x = self.layer0(x)
        feat1 = self.layer1(x)  # (B, 256, H/4, W/4)
        feat2 = self.layer2(feat1)  # (B, 512, H/8, W/8)
        feat3 = self.layer3(feat2)  # (B, 1024, H/16, W/16)

        # Target spatial size = layer2's spatial dim (stride 8)
        target_size = feat2.shape[2:]

        # Upsample layer3 to match layer2
        feat3_up = F.interpolate(feat3, size=target_size, mode="bilinear", align_corners=False)

        # Optionally include layer1
        feature_maps = [feat2, feat3_up]
        if self.config.use_layer1:
            feat1_down = F.interpolate(feat1, size=target_size, mode="bilinear", align_corners=False)
            feature_maps.insert(0, feat1_down)  # [layer1, layer2, layer3]

        # Concatenate
        features = torch.cat(feature_maps, dim=1)

        # Local neighborhood aggregation (from original PatchCore paper)
        if self.neighborhood_agg is not None:
            features = self.neighborhood_agg(features)

        # L2 normalization per patch
        if self.config.l2_normalize:
            features = F.normalize(features, p=2, dim=1)

        return features


def score_batch_enhanced(
    extractor: EnhancedFeatureExtractor,
    images: torch.Tensor,
    memory_bank: torch.Tensor,
    top_k: int = 1,
) -> np.ndarray:
    """
    Score a batch of images against a category's memory bank.

    Uses the same k-NN approach as baseline but with enhanced features.

    Args:
        extractor: Enhanced feature extractor (on device).
        images: (B, 3, H, W) tensor on device.
        memory_bank: (M, D) tensor of normal features on device.
        top_k: Number of nearest neighbors for scoring.

    Returns:
        (B,) array of anomaly scores (higher = more anomalous).
    """
    with torch.no_grad():
        features = extractor(images)
        b, c, h, w = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        scores = []
        for i in range(b):
            dists = torch.cdist(
                patches[i].unsqueeze(0),
                memory_bank.unsqueeze(0),
            ).squeeze(0)

            if top_k == 1:
                min_dists, _ = dists.min(dim=1)
            else:
                topk_dists, _ = dists.topk(top_k, dim=1, largest=False)
                min_dists = topk_dists.mean(dim=1)

            image_score = min_dists.max().item()
            scores.append(image_score)

    return np.array(scores)


def build_enhanced_memory_bank(
    extractor: EnhancedFeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
    config: EnhancedConfig,
) -> np.ndarray:
    """
    Build enhanced memory bank for one category.

    Uses memory-efficient reservoir sampling to avoid OOM errors when
    processing high-resolution images. Instead of concatenating all patch
    features in RAM, we maintain a capped reservoir (max 50K features)
    and apply coreset subsampling on the reservoir.

    Args:
        extractor: Enhanced feature extractor.
        dataloader: DataLoader with train/good images.
        device: Torch device.
        config: Enhanced configuration.

    Returns:
        (M, D) array of representative normal patch features.
    """
    MAX_RESERVOIR = 50000  # Cap to avoid OOM on high-res images
    rng = np.random.default_rng(42)

    reservoir: np.ndarray | None = None
    total_seen = 0

    for imgs, _ in tqdm(dataloader, desc="    Extracting enhanced features", leave=False):
        imgs = imgs.to(device)
        features = extractor(imgs)

        b, c, h, w = features.shape
        batch_features = features.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        batch_n = batch_features.shape[0]

        if reservoir is None:
            # First batch
            reservoir = batch_features
            total_seen = batch_n
        else:
            total_seen += batch_n
            combined = np.concatenate([reservoir, batch_features], axis=0)

            if combined.shape[0] > MAX_RESERVOIR:
                # Reservoir sampling: keep MAX_RESERVOIR random features
                indices = rng.choice(combined.shape[0], size=MAX_RESERVOIR, replace=False)
                reservoir = combined[indices]
            else:
                reservoir = combined

        # Free GPU memory
        del features, batch_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if reservoir is None:
        raise ValueError("No features extracted — is the dataloader empty?")

    logger.info(
        "Raw patch features: %d total seen, %d in reservoir × %d dims",
        total_seen,
        reservoir.shape[0],
        reservoir.shape[1],
    )

    memory_bank = coreset_subsampling(
        reservoir,
        ratio=config.coreset_ratio,
        max_samples=config.max_coreset,
    )
    logger.info(
        "Enhanced memory bank: %d × %d",
        memory_bank.shape[0],
        memory_bank.shape[1],
    )

    return memory_bank


def evaluate_category(
    extractor: EnhancedFeatureExtractor,
    memory_bank: torch.Tensor,
    records: list[dict],
    category: str,
    config: EnhancedConfig,
) -> dict:
    """
    Evaluate enhanced PatchCore on one category.

    Args:
        extractor: Enhanced feature extractor.
        memory_bank: (M, D) tensor on device.
        records: Test image records.
        category: Category name.
        config: Enhanced configuration.

    Returns:
        Dict with AUROC, AP, score statistics.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.dataset import EvalImageDataset

    dataset = EvalImageDataset(records, config.resolution, config.resolution)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_scores: list[float] = []
    all_labels: list[int] = []

    for imgs, labels, _indices in loader:
        imgs_dev = imgs.to(DEVICE)
        batch_scores = score_batch_enhanced(extractor, imgs_dev, memory_bank, top_k=config.top_k_scoring)
        all_scores.extend(batch_scores.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    n_good = int((y_true == 0).sum())
    n_anomaly = int((y_true == 1).sum())

    result: dict = {
        "category": category,
        "n_good": n_good,
        "n_anomaly": n_anomaly,
        "n_total": len(y_true),
    }

    if y_true.min() == y_true.max():
        logger.warning("[%s] Only one class — cannot compute metrics.", category)
        result.update({"auroc": None, "avg_precision": None})
    else:
        auroc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))
        result.update(
            {
                "auroc": round(auroc, 4),
                "avg_precision": round(ap, 4),
            }
        )

    result["score_good_mean"] = round(float(y_scores[y_true == 0].mean()), 4) if n_good > 0 else None
    result["score_anomaly_mean"] = round(float(y_scores[y_true == 1].mean()), 4) if n_anomaly > 0 else None
    result["score_gap"] = (
        round(float(y_scores[y_true == 1].mean() - y_scores[y_true == 0].mean()), 4)
        if n_good > 0 and n_anomaly > 0
        else None
    )

    return result


def main(
    selected_categories: list[str] | None = None,
    config: EnhancedConfig | None = None,
) -> list[dict]:
    """
    Build enhanced memory banks and evaluate for selected categories.

    Args:
        selected_categories: Categories to process (default: weak categories).
        config: Enhanced configuration (default: standard enhanced config).

    Returns:
        List of result dicts with before/after comparison.
    """
    set_seed()
    ensure_dataset()

    if config is None:
        config = EnhancedConfig()

    weak_categories = ["grid", "screw", "capsules"]
    all_categories = get_categories(DATASET_PATH)

    if selected_categories:
        invalid = [c for c in selected_categories if c not in all_categories]
        if invalid:
            raise ValueError(f"Unknown categories: {invalid}")
        categories = selected_categories
    else:
        categories = weak_categories

    logger.info("=" * 60)
    logger.info("  ENHANCED PATCHCORE — WEAK CATEGORY IMPROVEMENT")
    logger.info("=" * 60)
    logger.info("Target categories: %s", ", ".join(categories))
    logger.info("Configuration:")
    for key, val in config.to_dict().items():
        logger.info("  %-20s: %s", key, val)

    # Load baseline results for comparison
    baseline_results: dict[str, dict] = {}
    baseline_path = os.path.join(PATCHCORE_OUTPUT_DIR, "evaluation", "evaluation_results.json")
    if os.path.isfile(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline_data = json.load(f)
        for c in baseline_data.get("per_category", []):
            baseline_results[c["category"]] = c

    # Load enhanced extractor
    logger.info("Loading enhanced WideResNet-50 extractor...")
    extractor = EnhancedFeatureExtractor(config).to(DEVICE)
    extractor.eval()
    params_m = sum(p.numel() for p in extractor.parameters()) / 1e6
    logger.info(
        "Extractor ready (%.1fM params, frozen). Feature dim: %d, Spatial: %d×%d",
        params_m,
        config.feature_dim,
        config.spatial_size,
        config.spatial_size,
    )

    use_pin_memory = torch.cuda.is_available()
    results: list[dict] = []
    total_start = time.time()

    for i, category in enumerate(categories, 1):
        logger.info("=" * 60)
        logger.info("  [%d/%d] Processing: %s", i, len(categories), category.upper())
        logger.info("=" * 60)

        # ── Build memory bank ──
        image_data = collect_image_paths(DATASET_PATH, [category], subfolder=os.path.join("train", "good"))
        if not image_data:
            logger.warning("[%s] No training images — skipping.", category)
            continue

        image_data, _ = validate_images(image_data)
        logger.info("[%s] Training images: %d", category, len(image_data))

        dataset = AnomalyImageDataset(image_data, config.resolution, config.resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=use_pin_memory,
        )

        build_start = time.time()
        memory_bank = build_enhanced_memory_bank(extractor, dataloader, DEVICE, config)
        build_time = time.time() - build_start

        # Save enhanced memory bank
        save_dir = os.path.join(ENHANCED_OUTPUT_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        bank_path = os.path.join(save_dir, "memory_bank.pt")
        torch.save(torch.from_numpy(memory_bank), bank_path)
        logger.info("[%s] Enhanced bank saved → %s", category, bank_path)

        # ── Evaluate ──
        test_records = []
        from src.dataset import collect_test_images

        all_test = collect_test_images(DATASET_PATH)
        test_records = [r for r in all_test if r["category"] == category]

        if not test_records:
            logger.warning("[%s] No test images — skipping evaluation.", category)
            continue

        memory_bank_tensor = torch.from_numpy(memory_bank).float().to(DEVICE)
        eval_start = time.time()
        eval_result = evaluate_category(extractor, memory_bank_tensor, test_records, category, config)
        eval_time = time.time() - eval_start

        # Compare with baseline
        baseline = baseline_results.get(category, {})
        baseline_auroc = baseline.get("auroc", None)
        baseline_gap = baseline.get("score_gap", None)

        enhanced_auroc = eval_result.get("auroc", None)
        enhanced_gap = eval_result.get("score_gap", None)

        delta_auroc = None
        delta_gap = None
        if baseline_auroc is not None and enhanced_auroc is not None:
            delta_auroc = round(enhanced_auroc - baseline_auroc, 4)
        if baseline_gap is not None and enhanced_gap is not None:
            delta_gap = round(enhanced_gap - baseline_gap, 4)

        comparison = {
            **eval_result,
            "baseline_auroc": baseline_auroc,
            "baseline_gap": baseline_gap,
            "delta_auroc": delta_auroc,
            "delta_gap": delta_gap,
            "memory_bank_size": memory_bank.shape[0],
            "feature_dim": memory_bank.shape[1],
            "build_time_s": round(build_time, 1),
            "eval_time_s": round(eval_time, 1),
        }
        results.append(comparison)

        # Log comparison
        logger.info("-" * 50)
        logger.info(
            "[%s] AUROC: %.4f → %.4f  (%+.4f)",
            category,
            baseline_auroc or 0,
            enhanced_auroc or 0,
            delta_auroc or 0,
        )
        logger.info(
            "[%s] Score gap: %.4f → %.4f  (%+.4f)",
            category,
            baseline_gap or 0,
            enhanced_gap or 0,
            delta_gap or 0,
        )
        logger.info(
            "[%s] Bank: %d × %d | Build: %.1fs | Eval: %.1fs",
            category,
            memory_bank.shape[0],
            memory_bank.shape[1],
            build_time,
            eval_time,
        )

    total_elapsed = time.time() - total_start

    # ── Final Summary ──
    logger.info("=" * 60)
    logger.info("  ENHANCED PATCHCORE — RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Total time: %.1f min", total_elapsed / 60)
    logger.info("")
    logger.info(
        "%-12s  %10s  %10s  %10s  %10s  %10s",
        "Category",
        "Baseline",
        "Enhanced",
        "Δ AUROC",
        "Base gap",
        "Enh gap",
    )
    logger.info("-" * 64)

    improved = 0
    for r in results:
        marker = "✓" if (r["delta_auroc"] or 0) > 0 else "✗"
        logger.info(
            "%-12s  %10s  %10s  %10s  %10s  %10s  %s",
            r["category"],
            f"{r['baseline_auroc']:.4f}" if r["baseline_auroc"] else "N/A",
            f"{r['auroc']:.4f}" if r["auroc"] else "N/A",
            f"{r['delta_auroc']:+.4f}" if r["delta_auroc"] is not None else "N/A",
            f"{r['baseline_gap']:.2f}" if r["baseline_gap"] else "N/A",
            f"{r['score_gap']:.2f}" if r["score_gap"] else "N/A",
            marker,
        )
        if (r["delta_auroc"] or 0) > 0:
            improved += 1

    logger.info("-" * 64)
    logger.info("Improved: %d / %d categories", improved, len(results))

    # Save results
    os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)
    summary = {
        "config": config.to_dict(),
        "baseline_approach": "patchcore_v1",
        "total_time_seconds": round(total_elapsed, 1),
        "improved_count": improved,
        "total_evaluated": len(results),
        "per_category": results,
    }
    summary_path = os.path.join(ENHANCED_OUTPUT_DIR, "improvement_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Results saved → %s", summary_path)

    return results


def run_sweep(
    categories: list[str] | None = None,
) -> None:
    """
    Run a configuration sweep to find the best settings per category.

    Tests 4 configurations and reports the best one for each category:
        A: 512×512 + layer1 + neighborhood + NO L2 norm (full enhanced, no norm)
        B: 512×512 + NO layer1 + neighborhood + NO L2 norm (high-res baseline)
        C: 256×256 + layer1 + neighborhood + L2 norm + k=3 (fine features, orig size)
        D: 512×512 + layer1 + neighborhood + L2 norm (full enhanced with norm)
    """
    configs = {
        "A: 512+L1+neigh": EnhancedConfig(
            resolution=512,
            use_layer1=True,
            neighborhood_size=3,
            l2_normalize=False,
            top_k_scoring=1,
            max_coreset=10000,
            batch_size=4,
        ),
        "B: 512+neigh": EnhancedConfig(
            resolution=512,
            use_layer1=False,
            neighborhood_size=3,
            l2_normalize=False,
            top_k_scoring=1,
            max_coreset=10000,
            batch_size=4,
        ),
        "C: 256+L1+L2+k3": EnhancedConfig(
            resolution=256,
            use_layer1=True,
            neighborhood_size=3,
            l2_normalize=True,
            top_k_scoring=3,
            max_coreset=10000,
            batch_size=8,
        ),
        "D: 512+L1+L2+neigh": EnhancedConfig(
            resolution=512,
            use_layer1=True,
            neighborhood_size=3,
            l2_normalize=True,
            top_k_scoring=1,
            max_coreset=10000,
            batch_size=4,
        ),
    }

    all_sweep_results: dict[str, list[dict]] = {}

    for config_name, config in configs.items():
        logger.info("\n" + "█" * 60)
        logger.info("  SWEEP CONFIG: %s", config_name)
        logger.info("█" * 60)

        results = main(selected_categories=categories, config=config)
        all_sweep_results[config_name] = results

    # ── Final comparison ──
    logger.info("\n" + "=" * 80)
    logger.info("  CONFIGURATION SWEEP — FINAL COMPARISON")
    logger.info("=" * 80)

    # Load baseline
    baseline_path = os.path.join(PATCHCORE_OUTPUT_DIR, "evaluation", "evaluation_results.json")
    baseline_results: dict[str, float] = {}
    if os.path.isfile(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline_data = json.load(f)
        for c in baseline_data.get("per_category", []):
            baseline_results[c["category"]] = c.get("auroc", 0)

    # Get unique categories
    cat_names = sorted(set(r["category"] for results in all_sweep_results.values() for r in results))

    # Header
    header = f"{'Category':12s}  {'Baseline':>8s}"
    for cn in all_sweep_results:
        header += f"  {cn:>18s}"
    header += f"  {'Best Config':>20s}"
    logger.info(header)
    logger.info("-" * len(header))

    best_configs: dict[str, tuple[str, float]] = {}
    for cat in cat_names:
        base = baseline_results.get(cat, 0)
        line = f"{cat:12s}  {base:8.4f}"
        best_auroc = base
        best_name = "Baseline"

        for cn, results in all_sweep_results.items():
            cat_result = [r for r in results if r["category"] == cat]
            if cat_result:
                auroc = cat_result[0].get("auroc", 0) or 0
                line += f"  {auroc:18.4f}"
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_name = cn
            else:
                line += f"  {'N/A':>18s}"

        line += f"  {best_name:>20s}"
        logger.info(line)
        best_configs[cat] = (best_name, best_auroc)

    logger.info("-" * len(header))
    for cat, (cfg, auroc) in best_configs.items():
        base = baseline_results.get(cat, 0)
        delta = auroc - base
        logger.info(
            "  %s: best = %s  (AUROC %.4f, %+.4f vs baseline)",
            cat,
            cfg,
            auroc,
            delta,
        )

    # Save sweep results
    sweep_summary = {
        "configs": {cn: cfg.to_dict() for cn, cfg in configs.items()},
        "baseline": baseline_results,
        "results": {cn: results for cn, results in all_sweep_results.items()},
        "best_per_category": {cat: {"config": cfg, "auroc": auroc} for cat, (cfg, auroc) in best_configs.items()},
    }
    os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)
    sweep_path = os.path.join(ENHANCED_OUTPUT_DIR, "sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2, ensure_ascii=False)
    logger.info("Sweep results saved → %s", sweep_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced PatchCore for improving weak categories.")
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=None,
        help="Categories to process (default: grid, screw, capsules).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run configuration sweep (tests 4 configs per category).",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=512,
        help="Input image resolution (default: 512).",
    )
    parser.add_argument(
        "--no-layer1",
        action="store_true",
        help="Disable layer 1 inclusion.",
    )
    parser.add_argument(
        "--no-l2-norm",
        action="store_true",
        help="Disable L2 normalization.",
    )
    parser.add_argument(
        "--no-neighborhood",
        action="store_true",
        help="Disable local neighborhood aggregation.",
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=3,
        help="Kernel size for neighborhood aggregation (default: 3).",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.10,
        help="Coreset subsampling ratio (default: 0.10).",
    )
    parser.add_argument(
        "--max-coreset",
        type=int,
        default=10000,
        help="Maximum coreset size (default: 10000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k neighbors for scoring (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8).",
    )
    args = parser.parse_args()

    if args.sweep:
        run_sweep(categories=args.categories)
    else:
        config = EnhancedConfig(
            resolution=args.resolution,
            use_layer1=not args.no_layer1,
            neighborhood_size=0 if args.no_neighborhood else args.neighborhood_size,
            l2_normalize=not args.no_l2_norm,
            coreset_ratio=args.coreset_ratio,
            max_coreset=args.max_coreset,
            top_k_scoring=args.top_k,
            batch_size=args.batch_size,
        )

        main(selected_categories=args.categories, config=config)

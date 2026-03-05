"""
Apply Best Enhanced PatchCore Configurations.

After running the configuration sweep, this script applies the winning
configuration for each weak category and rebuilds their memory banks
using the enhanced feature extraction pipeline.

For categories where the enhanced config outperforms the baseline,
the enhanced memory bank replaces the original one in the main
PatchCore output directory.

Usage:
    python -m src.models.patchcore.apply_best_configs
    python -m src.models.patchcore.apply_best_configs --dry-run
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.config import DATASET_PATH, DEVICE, NUM_WORKERS, ensure_dataset, set_seed
from src.dataset import (
    AnomalyImageDataset,
    collect_image_paths,
    collect_test_images,
    validate_images,
)
from src.logger import get_logger
from src.models.patchcore.build_memory_bank import PATCHCORE_OUTPUT_DIR
from src.models.patchcore.enhanced_features import (
    ENHANCED_OUTPUT_DIR,
    EnhancedConfig,
    EnhancedFeatureExtractor,
    build_enhanced_memory_bank,
    evaluate_category,
)

logger = get_logger(__name__)


BEST_CONFIGS: dict[str, EnhancedConfig] = {
    "grid": EnhancedConfig(
        resolution=512,
        use_layer1=False,
        neighborhood_size=3,
        l2_normalize=False,
        coreset_ratio=0.10,
        max_coreset=10000,
        top_k_scoring=1,
        batch_size=4,
    ),
    "capsules": EnhancedConfig(
        resolution=512,
        use_layer1=False,
        neighborhood_size=3,
        l2_normalize=False,
        coreset_ratio=0.10,
        max_coreset=10000,
        top_k_scoring=1,
        batch_size=4,
    ),
    "screw": EnhancedConfig(
        resolution=512,
        use_layer1=True,
        neighborhood_size=3,
        l2_normalize=True,
        coreset_ratio=0.10,
        max_coreset=10000,
        top_k_scoring=1,
        batch_size=4,
    ),
}
"""Winning configurations from the sweep for each weak category."""


def apply_best_configs(
    categories: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """
    Apply the best enhanced configurations to weak categories.

    Rebuilds memory banks with the enhanced extractor and evaluates.
    If improved, copies the enhanced bank to the main PatchCore directory.

    Args:
        categories: Categories to process (default: all weak categories).
        dry_run: If True, only evaluate without replacing memory banks.

    Returns:
        Dict mapping category → result dict.
    """
    set_seed()
    ensure_dataset()

    if categories is None:
        categories = list(BEST_CONFIGS.keys())

    # Load baseline results
    baseline_results: dict[str, float] = {}
    baseline_path = os.path.join(PATCHCORE_OUTPUT_DIR, "evaluation", "evaluation_results.json")
    if os.path.isfile(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline_data = json.load(f)
        for c in baseline_data.get("per_category", []):
            baseline_results[c["category"]] = c.get("auroc", 0)

    results: dict[str, dict] = {}
    for category in categories:
        if category not in BEST_CONFIGS:
            logger.warning("[%s] No best config found — skipping.", category)
            continue

        config = BEST_CONFIGS[category]
        baseline_auroc = baseline_results.get(category, 0)

        logger.info("=" * 60)
        logger.info("  [%s] Applying best config", category.upper())
        logger.info("=" * 60)
        logger.info(
            "  Config: resolution=%d, layer1=%s, L2=%s, neighborhood=%d",
            config.resolution,
            config.use_layer1,
            config.l2_normalize,
            config.neighborhood_size,
        )
        logger.info("  Baseline AUROC: %.4f", baseline_auroc)

        # Build enhanced memory bank
        extractor = EnhancedFeatureExtractor(config).to(DEVICE)
        extractor.eval()

        image_data = collect_image_paths(DATASET_PATH, [category], subfolder=os.path.join("train", "good"))
        image_data, _ = validate_images(image_data)

        dataset = AnomalyImageDataset(image_data, config.resolution, config.resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        memory_bank = build_enhanced_memory_bank(extractor, dataloader, DEVICE, config)

        # Save enhanced bank
        save_dir = os.path.join(ENHANCED_OUTPUT_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        bank_path = os.path.join(save_dir, "memory_bank.pt")
        torch.save(torch.from_numpy(memory_bank), bank_path)

        # Evaluate
        all_test = collect_test_images(DATASET_PATH)
        test_records = [r for r in all_test if r["category"] == category]
        memory_bank_tensor = torch.from_numpy(memory_bank).float().to(DEVICE)

        eval_result = evaluate_category(extractor, memory_bank_tensor, test_records, category, config)

        enhanced_auroc = eval_result.get("auroc", 0)
        delta = enhanced_auroc - baseline_auroc

        logger.info("  Enhanced AUROC: %.4f (%+.4f)", enhanced_auroc, delta)

        result = {
            "category": category,
            "baseline_auroc": baseline_auroc,
            "enhanced_auroc": enhanced_auroc,
            "delta": round(delta, 4),
            "improved": delta > 0,
            "config": config.to_dict(),
        }

        if delta > 0 and not dry_run:
            # Save config alongside the memory bank
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2)

            logger.info("  ✓ Memory bank saved to enhanced directory.")
            result["applied"] = True
        elif dry_run:
            logger.info("  [DRY RUN] Would apply enhanced config.")
            result["applied"] = False
        else:
            logger.info("  ✗ Enhanced config did not improve — keeping baseline.")
            result["applied"] = False

        results[category] = result

        # Free GPU memory
        del extractor, memory_bank_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary = {
        "action": "apply_best_configs",
        "dry_run": dry_run,
        "results": results,
    }
    summary_path = os.path.join(ENHANCED_OUTPUT_DIR, "applied_configs.json")
    os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved → %s", summary_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply best enhanced PatchCore configs to weak categories.")
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=None,
        help="Categories to apply (default: grid, screw, capsules).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only evaluate, do not replace memory banks.",
    )
    args = parser.parse_args()

    apply_best_configs(
        categories=args.categories,
        dry_run=args.dry_run,
    )

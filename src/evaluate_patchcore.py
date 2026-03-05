"""
PatchCore evaluation pipeline.

Loads each category's memory bank and scores test images using
nearest-neighbor distance in WideResNet-50 feature space.
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
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_WORKERS,
    OUTPUTS_DIR,
    set_seed,
)
from src.dataset import EvalImageDataset, collect_test_images
from src.logger import get_logger
from src.models.patchcore.build_memory_bank import (
    PATCHCORE_OUTPUT_DIR,
    PatchCoreFeatureExtractor,
)

logger = get_logger(__name__)

EVAL_OUTPUT_DIR = os.path.join(PATCHCORE_OUTPUT_DIR, "evaluation")


def score_batch(
    extractor: PatchCoreFeatureExtractor,
    images: torch.Tensor,
    memory_bank: torch.Tensor,
    top_k: int = 3,
) -> np.ndarray:
    """
    Score a batch of images against a category's memory bank.

    For each patch feature in the test image, finds the nearest neighbor
    in the memory bank. The image-level anomaly score is the maximum
    patch-level distance (identifies the most anomalous patch).

    Args:
        extractor: Frozen WideResNet-50 feature extractor.
        images: (B, 3, H, W) tensor on device.
        memory_bank: (M, D) tensor of normal features on device.
        top_k: Number of nearest neighbors to average for patch score.

    Returns:
        (B,) array of anomaly scores (higher = more anomalous).
    """
    with torch.no_grad():
        features = extractor(images)  # (B, 1536, H', W')
        b, c, h, w = features.shape

        # Reshape to (B, H'*W', 1536)
        patches = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        scores = []
        for i in range(b):
            # (H'*W', 1536) vs (M, 1536) → (H'*W', M)
            dists = torch.cdist(patches[i].unsqueeze(0), memory_bank.unsqueeze(0)).squeeze(0)

            # For each patch, get distance to k-nearest neighbors
            if top_k == 1:
                min_dists, _ = dists.min(dim=1)  # (H'*W',)
            else:
                topk_dists, _ = dists.topk(top_k, dim=1, largest=False)
                min_dists = topk_dists.mean(dim=1)  # (H'*W',)

            # Image-level score: max over all patches
            image_score = min_dists.max().item()
            scores.append(image_score)

    return np.array(scores)


def evaluate_single_category(
    extractor: PatchCoreFeatureExtractor,
    memory_bank: torch.Tensor,
    records: list[dict],
    category: str,
) -> dict:
    """
    Evaluate PatchCore on one category's test images.

    Args:
        extractor: Feature extractor (on device).
        memory_bank: (M, D) tensor of normal features (on device).
        records: Test image records for this category.
        category: Category name.

    Returns:
        Dict with metrics and scores.
    """
    dataset = EvalImageDataset(records, IMG_HEIGHT, IMG_WIDTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_scores: list[float] = []
    all_labels: list[int] = []

    for imgs, labels, _indices in loader:
        imgs_dev = imgs.to(DEVICE)
        batch_scores = score_batch(extractor, imgs_dev, memory_bank)
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
        logger.warning("[%s] Only one class — cannot compute AUROC.", category)
        result.update(
            {
                "auroc": None,
                "avg_precision": None,
            }
        )
    else:
        auroc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))
        result.update(
            {
                "auroc": round(auroc, 4),
                "avg_precision": round(ap, 4),
            }
        )

    # Score statistics
    result["score_good_mean"] = round(float(y_scores[y_true == 0].mean()), 4) if n_good > 0 else None
    result["score_anomaly_mean"] = round(float(y_scores[y_true == 1].mean()), 4) if n_anomaly > 0 else None
    result["score_gap"] = (
        round(float(y_scores[y_true == 1].mean() - y_scores[y_true == 0].mean()), 4)
        if n_good > 0 and n_anomaly > 0
        else None
    )

    # Return raw scores and labels for aggregated metrics (avoids re-scoring)
    result["_raw_labels"] = all_labels
    result["_raw_scores"] = all_scores

    return result


def evaluate_all(
    compare_global_model: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate PatchCore on all categories with available memory banks.

    Args:
        compare_global_model: Name from MODEL_REGISTRY to compare against.

    Returns:
        DataFrame with per-category metrics.
    """
    set_seed()
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # Collect test images
    all_records = collect_test_images(DATASET_PATH)
    categories = sorted(set(r["category"] for r in all_records))
    logger.info("Test images: %d across %d categories", len(all_records), len(categories))

    records_by_cat: dict[str, list[dict]] = {}
    for r in all_records:
        records_by_cat.setdefault(r["category"], []).append(r)

    # Load backbone
    logger.info("Loading WideResNet-50 backbone...")
    extractor = PatchCoreFeatureExtractor().to(DEVICE)
    extractor.eval()
    logger.info("Backbone ready.")

    # Evaluate each category
    results: list[dict] = []
    all_labels_agg: list[int] = []
    all_scores_agg: list[float] = []

    for i, category in enumerate(categories, 1):
        logger.info("[%d/%d] Evaluating PatchCore: %s", i, len(categories), category.upper())

        # Load memory bank
        bank_path = os.path.join(PATCHCORE_OUTPUT_DIR, category, "memory_bank.pt")
        if not os.path.isfile(bank_path):
            logger.warning("[%s] No memory bank found — skipping.", category)
            continue

        memory_bank = torch.load(bank_path, map_location=DEVICE, weights_only=True)
        if not isinstance(memory_bank, torch.Tensor):
            memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
        memory_bank = memory_bank.to(DEVICE)
        logger.info("[%s] Memory bank: %d × %d", category, *memory_bank.shape)

        cat_records = records_by_cat.get(category, [])
        if not cat_records:
            logger.warning("[%s] No test images.", category)
            continue

        result = evaluate_single_category(extractor, memory_bank, cat_records, category)
        results.append(result)

        all_labels_agg.extend(result.pop("_raw_labels"))
        all_scores_agg.extend(result.pop("_raw_scores"))

        auroc_str = f"{result['auroc']:.4f}" if result["auroc"] is not None else "N/A"
        logger.info(
            "  %s  AUROC: %s  (good=%d, anom=%d)",
            category,
            auroc_str,
            result["n_good"],
            result["n_anomaly"],
        )

    df = pd.DataFrame(results)

    valid = df.dropna(subset=["auroc"])

    if len(valid) > 0:
        mean_auroc = valid["auroc"].mean()
        median_auroc = valid["auroc"].median()

        # Weighted mean
        valid_w = valid.copy()
        valid_w["weight"] = valid_w["n_total"]
        weighted_auroc = (valid_w["auroc"] * valid_w["weight"]).sum() / valid_w["weight"].sum()

        # Global AUROC from aggregated scores
        if len(set(all_labels_agg)) > 1:
            global_auroc = roc_auc_score(all_labels_agg, all_scores_agg)
            global_ap = average_precision_score(all_labels_agg, all_scores_agg)

            # Optimal threshold
            fpr, tpr, thresholds = roc_curve(all_labels_agg, all_scores_agg)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]
            y_pred = (np.array(all_scores_agg) >= best_threshold).astype(int)
            cm = confusion_matrix(all_labels_agg, y_pred)
        else:
            global_auroc = None
            global_ap = None
            best_threshold = None
            cm = None

        logger.info("=" * 60)
        logger.info("  PATCHCORE EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info("Categories evaluated: %d / %d", len(valid), len(categories))
        logger.info("Mean AUROC:     %.4f", mean_auroc)
        logger.info("Median AUROC:   %.4f", median_auroc)
        logger.info("Weighted AUROC: %.4f", weighted_auroc)
        if global_auroc:
            logger.info("Global AUROC (aggregated): %.4f", global_auroc)
            logger.info("Global AP (aggregated):    %.4f", global_ap)
            logger.info("Optimal threshold: %.4f", best_threshold)
            logger.info("Confusion Matrix:\n%s", cm)
            logger.info(
                "Classification Report:\n%s",
                classification_report(all_labels_agg, y_pred, target_names=["good", "anomaly"]),
            )
    else:
        mean_auroc = None
        median_auroc = None
        weighted_auroc = None
        global_auroc = None
        global_ap = None
        best_threshold = None
        cm = None

    global_results = None
    if compare_global_model:
        global_json = os.path.join(OUTPUTS_DIR, compare_global_model, "evaluation", "evaluation_results.json")
        if os.path.isfile(global_json):
            with open(global_json, encoding="utf-8") as f:
                global_results = json.load(f)
            logger.info("Loaded global model results for comparison.")

    _generate_charts(df, global_results, compare_global_model)

    df.to_csv(os.path.join(EVAL_OUTPUT_DIR, "patchcore_metrics.csv"), index=False)

    summary = {
        "approach": "patchcore",
        "backbone": "wide_resnet50_2",
        "total_categories_evaluated": len(valid) if valid is not None else 0,
        "mean_auroc": round(float(mean_auroc), 4) if mean_auroc else None,
        "median_auroc": round(float(median_auroc), 4) if median_auroc else None,
        "weighted_auroc": round(float(weighted_auroc), 4) if weighted_auroc else None,
        "global_auroc_aggregated": round(float(global_auroc), 4) if global_auroc else None,
        "global_ap_aggregated": round(float(global_ap), 4) if global_ap else None,
        "optimal_threshold": round(float(best_threshold), 4) if best_threshold else None,
        "per_category": results,
    }

    json_path = os.path.join(EVAL_OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", EVAL_OUTPUT_DIR)
    return df


def _generate_charts(
    df: pd.DataFrame,
    global_results: dict | None,
    global_model_name: str | None,
) -> None:
    """Generate PatchCore evaluation charts."""

    valid = df.dropna(subset=["auroc"]).sort_values("auroc", ascending=True)
    if len(valid) == 0:
        return

    # AUROC bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.6 else "#e74c3c" for v in valid["auroc"]]
    bars = ax.barh(valid["category"], valid["auroc"], color=colors, edgecolor="black")
    ax.axvline(0.5, color="gray", linestyle=":", lw=1, label="Random (0.5)")

    mean_auroc = valid["auroc"].mean()
    ax.axvline(mean_auroc, color="blue", linestyle="--", lw=2, label=f"Mean = {mean_auroc:.3f}")

    if global_results and "global_metrics" in global_results:
        g_auroc = global_results["global_metrics"].get("auroc_combined")
        if g_auroc:
            ax.axvline(g_auroc, color="red", linestyle="-.", lw=2, label=f"Global {global_model_name} = {g_auroc:.3f}")

    for bar, val in zip(bars, valid["auroc"], strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("AUROC")
    ax.set_title("PatchCore – AUROC by Category", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, "auroc_per_category.png"), dpi=150, bbox_inches="tight")
    logger.info("  → auroc_per_category.png")

    # Comparison with global model
    if global_results and "per_category" in global_results:
        global_cats = {r["category"]: r.get("auroc", r.get("auroc_mse")) for r in global_results["per_category"]}

        compare_data = []
        for _, row in valid.iterrows():
            cat = row["category"]
            g = global_cats.get(cat)
            if g is not None:
                compare_data.append(
                    {
                        "category": cat,
                        "patchcore": row["auroc"],
                        "global": g,
                        "improvement": row["auroc"] - g,
                    }
                )

        if compare_data:
            df_cmp = pd.DataFrame(compare_data).sort_values("improvement", ascending=True)

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            x = np.arange(len(df_cmp))
            width = 0.35
            axes[0].barh(
                x - width / 2,
                df_cmp["global"],
                width,
                label=f"Global ({global_model_name})",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
            )
            axes[0].barh(
                x + width / 2,
                df_cmp["patchcore"],
                width,
                label="PatchCore",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
            )
            axes[0].set_yticks(x)
            axes[0].set_yticklabels(df_cmp["category"])
            axes[0].set_xlabel("AUROC")
            axes[0].set_title("Global AE vs PatchCore", fontsize=13, fontweight="bold")
            axes[0].axvline(0.5, color="gray", linestyle=":", lw=1)
            axes[0].legend(loc="lower right")
            axes[0].grid(True, alpha=0.3, axis="x")
            axes[0].set_xlim(0, 1.15)

            colors_imp = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_cmp["improvement"]]
            axes[1].barh(df_cmp["category"], df_cmp["improvement"], color=colors_imp, edgecolor="black")
            axes[1].axvline(0, color="black", lw=1)
            axes[1].set_xlabel("AUROC Improvement (PatchCore − Global)")
            axes[1].set_title("PatchCore Improvement", fontsize=13, fontweight="bold")
            axes[1].grid(True, alpha=0.3, axis="x")

            for bar, val in zip(axes[1].patches, df_cmp["improvement"], strict=False):
                x_pos = bar.get_width() + (0.01 if val >= 0 else -0.01)
                ha = "left" if val >= 0 else "right"
                axes[1].text(
                    x_pos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    ha=ha,
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            plt.tight_layout()
            plt.savefig(
                os.path.join(EVAL_OUTPUT_DIR, "comparison_patchcore_vs_global.png"), dpi=150, bbox_inches="tight"
            )
            logger.info("  → comparison_patchcore_vs_global.png")

            df_cmp.to_csv(os.path.join(EVAL_OUTPUT_DIR, "comparison_patchcore_vs_global.csv"), index=False)

            improved = sum(1 for v in df_cmp["improvement"] if v > 0)
            degraded = sum(1 for v in df_cmp["improvement"] if v < 0)
            logger.info(
                "Comparison: %d improved, %d degraded, avg: %+.4f", improved, degraded, df_cmp["improvement"].mean()
            )

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore anomaly detection.")
    parser.add_argument(
        "--compare-with",
        "-cw",
        type=str,
        default="autoencoder",
        help="Global model to compare against (default: autoencoder). Use 'none' to skip.",
    )
    args = parser.parse_args()
    compare = args.compare_with if args.compare_with.lower() != "none" else None
    evaluate_all(compare_global_model=compare)


if __name__ == "__main__":
    main()

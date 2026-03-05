"""
Per-Category Autoencoder Evaluation Pipeline.

Loads each category's specialized model and evaluates it on that
category's test images. Compares results with the global model.
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
    roc_auc_score,
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
from src.feature_extractor import VGGFeatureExtractor, compute_perceptual_score
from src.logger import get_logger
from src.metrics import compute_combined_score, compute_ssim_batch
from src.models.autoencoder import Autoencoder

logger = get_logger(__name__)

PER_CATEGORY_MODEL_DIR = os.path.join(OUTPUTS_DIR, "autoencoder_per_category")
EVAL_OUTPUT_DIR = os.path.join(PER_CATEGORY_MODEL_DIR, "evaluation")


def load_category_model(category: str) -> torch.nn.Module | None:
    """
    Load the trained per-category Autoencoder.

    Args:
        category: Category name (e.g. 'bottle').

    Returns:
        Loaded model in eval mode, or None if weights not found.
    """
    weights_path = os.path.join(PER_CATEGORY_MODEL_DIR, category, "model.pth")
    if not os.path.isfile(weights_path):
        logger.warning("[%s] No trained model found at %s", category, weights_path)
        return None

    model = Autoencoder()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_single_category(
    model: torch.nn.Module,
    records: list[dict],
    vgg_extractor: VGGFeatureExtractor,
    category: str,
) -> dict:
    """
    Evaluate a per-category model on its test images.

    Args:
        model: Trained Autoencoder (in eval mode, on DEVICE).
        records: Test image records for this category.
        vgg_extractor: Frozen VGG feature extractor (on DEVICE).
        category: Category name.

    Returns:
        Dict with per-category metrics and per-image scores.
    """
    dataset = EvalImageDataset(records, IMG_HEIGHT, IMG_WIDTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_mse: list[float] = []
    all_ssim: list[float] = []
    all_perceptual: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, labels, _indices in loader:
            imgs_dev = imgs.to(DEVICE)
            preds = model(imgs_dev)

            mse = ((imgs_dev - preds) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            ssim = compute_ssim_batch(imgs_dev, preds).cpu().numpy()
            perceptual = compute_perceptual_score(vgg_extractor, imgs_dev, preds).cpu().numpy()

            all_mse.extend(mse.tolist())
            all_ssim.extend(ssim.tolist())
            all_perceptual.extend(perceptual.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    scores_mse = np.array(all_mse)
    scores_ssim = 1 - np.array(all_ssim)
    scores_perceptual = np.array(all_perceptual)

    scores_combined = compute_combined_score(scores_mse, scores_ssim, scores_perceptual)

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
                "auroc_mse": None,
                "auroc_ssim": None,
                "auroc_perceptual": None,
                "auroc_combined": None,
                "avg_precision": None,
            }
        )
    else:
        result.update(
            {
                "auroc_mse": round(float(roc_auc_score(y_true, scores_mse)), 4),
                "auroc_ssim": round(float(roc_auc_score(y_true, scores_ssim)), 4),
                "auroc_perceptual": round(float(roc_auc_score(y_true, scores_perceptual)), 4),
                "auroc_combined": round(float(roc_auc_score(y_true, scores_combined)), 4),
                "avg_precision": round(float(average_precision_score(y_true, scores_combined)), 4),
            }
        )

    # Error statistics
    result["mse_good_mean"] = round(float(scores_mse[y_true == 0].mean()), 6) if n_good > 0 else None
    result["mse_anomaly_mean"] = round(float(scores_mse[y_true == 1].mean()), 6) if n_anomaly > 0 else None

    # Raw scores for global aggregation (avoids re-inference)
    result["_raw_labels"] = all_labels
    result["_raw_scores"] = scores_combined.tolist()

    return result


def evaluate_all(
    compare_global_model: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate all per-category models and optionally compare with a global model.

    Args:
        compare_global_model: Name of global model to compare against
                              (e.g. 'autoencoder'). If None, skip comparison.

    Returns:
        DataFrame with per-category metrics.
    """
    set_seed()
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # Collect all test images
    all_records = collect_test_images(DATASET_PATH)
    categories = sorted(set(r["category"] for r in all_records))
    logger.info("Test images: %d across %d categories", len(all_records), len(categories))

    # Group records by category
    records_by_cat: dict[str, list[dict]] = {}
    for r in all_records:
        records_by_cat.setdefault(r["category"], []).append(r)

    # Load VGG extractor
    logger.info("Loading VGG-16 feature extractor...")
    vgg_extractor = VGGFeatureExtractor().to(DEVICE)
    vgg_extractor.eval()

    # Evaluate each category
    results: list[dict] = []
    all_labels_agg: list[int] = []
    all_scores_agg: list[float] = []

    for i, category in enumerate(categories, 1):
        logger.info("[%d/%d] Evaluating: %s", i, len(categories), category.upper())

        model = load_category_model(category)
        if model is None:
            logger.warning("[%s] Skipped (no model).", category)
            continue

        cat_records = records_by_cat.get(category, [])
        if not cat_records:
            logger.warning("[%s] No test images found.", category)
            continue

        result = evaluate_single_category(model, cat_records, vgg_extractor, category)

        # Collect raw scores for global aggregation (avoids re-inference)
        all_labels_agg.extend(result.pop("_raw_labels"))
        all_scores_agg.extend(result.pop("_raw_scores"))

        results.append(result)

        auroc_str = f"{result['auroc_combined']:.4f}" if result["auroc_combined"] is not None else "N/A"
        logger.info(
            "  %s  AUROC(combined): %s  (good=%d, anom=%d)",
            category,
            auroc_str,
            result["n_good"],
            result["n_anomaly"],
        )

    df = pd.DataFrame(results)

    valid = df.dropna(subset=["auroc_combined"])
    if len(valid) > 0:
        mean_auroc = valid["auroc_combined"].mean()
        median_auroc = valid["auroc_combined"].median()
        valid_with_weight = valid.copy()
        valid_with_weight["weight"] = valid_with_weight["n_total"]
        weighted_auroc = (valid_with_weight["auroc_combined"] * valid_with_weight["weight"]).sum() / valid_with_weight[
            "weight"
        ].sum()

        if len(set(all_labels_agg)) > 1:
            global_auroc = roc_auc_score(all_labels_agg, all_scores_agg)
            global_ap = average_precision_score(all_labels_agg, all_scores_agg)
        else:
            global_auroc = None
            global_ap = None

        logger.info("=" * 60)
        logger.info("  PER-CATEGORY EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info("Categories evaluated: %d / %d", len(valid), len(categories))
        logger.info("Mean AUROC (combined): %.4f", mean_auroc)
        logger.info("Median AUROC:          %.4f", median_auroc)
        logger.info("Weighted AUROC:        %.4f", weighted_auroc)
        if global_auroc is not None:
            logger.info("Global AUROC (aggregated): %.4f", global_auroc)
            logger.info("Global AP (aggregated):    %.4f", global_ap)
    else:
        mean_auroc = None
        median_auroc = None
        weighted_auroc = None
        global_auroc = None
        global_ap = None

    global_results = None
    if compare_global_model:
        global_json_path = os.path.join(OUTPUTS_DIR, compare_global_model, "evaluation", "evaluation_results.json")
        if os.path.isfile(global_json_path):
            with open(global_json_path, encoding="utf-8") as f:
                global_results = json.load(f)
            logger.info("Loaded global model results from: %s", global_json_path)
        else:
            logger.warning(
                "Global model results not found at %s. Run `python src/evaluate.py --model %s` first.",
                global_json_path,
                compare_global_model,
            )

    _generate_charts(df, global_results, compare_global_model)

    df.to_csv(os.path.join(EVAL_OUTPUT_DIR, "per_category_metrics.csv"), index=False)
    logger.info("  → per_category_metrics.csv")

    summary = {
        "approach": "autoencoder_per_category",
        "total_categories_evaluated": len(valid) if valid is not None else 0,
        "mean_auroc_combined": round(float(mean_auroc), 4) if mean_auroc else None,
        "median_auroc_combined": round(float(median_auroc), 4) if median_auroc else None,
        "weighted_auroc_combined": round(float(weighted_auroc), 4) if weighted_auroc else None,
        "global_auroc_aggregated": round(float(global_auroc), 4) if global_auroc else None,
        "global_ap_aggregated": round(float(global_ap), 4) if global_ap else None,
        "per_category": results,
    }

    json_path = os.path.join(EVAL_OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  → evaluation_results.json")
    logger.info("All results saved to: %s", EVAL_OUTPUT_DIR)

    return df


def _generate_charts(
    df: pd.DataFrame,
    global_results: dict | None,
    global_model_name: str | None,
) -> None:
    """Generate comparison and summary charts."""

    valid = df.dropna(subset=["auroc_combined"]).sort_values("auroc_combined", ascending=True)
    if len(valid) == 0:
        logger.warning("No valid AUROC values to plot.")
        return

    # Per-category AUROC bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.6 else "#e74c3c" for v in valid["auroc_combined"]]
    bars = ax.barh(valid["category"], valid["auroc_combined"], color=colors, edgecolor="black")
    ax.axvline(0.5, color="gray", linestyle=":", lw=1, label="Random (0.5)")

    mean_auroc = valid["auroc_combined"].mean()
    ax.axvline(
        mean_auroc,
        color="blue",
        linestyle="--",
        lw=2,
        label=f"Mean AUROC = {mean_auroc:.3f}",
    )

    # Add global model reference line
    if global_results and "global_metrics" in global_results:
        global_auroc = global_results["global_metrics"].get("auroc_combined", None)
        if global_auroc:
            ax.axvline(
                global_auroc,
                color="red",
                linestyle="-.",
                lw=2,
                label=f"Global {global_model_name} = {global_auroc:.3f}",
            )

    for bar, val in zip(bars, valid["auroc_combined"], strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("AUROC (Combined)")
    ax.set_title("Per-Category Autoencoder – AUROC by Category", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_OUTPUT_DIR, "auroc_per_category.png"),
        dpi=150,
        bbox_inches="tight",
    )
    logger.info("  → auroc_per_category.png")

    # Comparison with global model
    if global_results and "per_category" in global_results:
        global_cats = {r["category"]: r["auroc"] for r in global_results["per_category"]}

        compare_data = []
        for _, row in valid.iterrows():
            cat = row["category"]
            per_cat_auroc = row["auroc_combined"]
            global_auroc = global_cats.get(cat)
            if global_auroc is not None:
                compare_data.append(
                    {
                        "category": cat,
                        "per_category": per_cat_auroc,
                        "global": global_auroc,
                        "improvement": per_cat_auroc - global_auroc,
                    }
                )

        if compare_data:
            df_cmp = pd.DataFrame(compare_data).sort_values("improvement", ascending=True)

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # Side-by-side bar chart
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
                df_cmp["per_category"],
                width,
                label="Per-Category AE",
                color="#2ecc71",
                alpha=0.8,
                edgecolor="black",
            )
            axes[0].set_yticks(x)
            axes[0].set_yticklabels(df_cmp["category"])
            axes[0].set_xlabel("AUROC (Combined)")
            axes[0].set_title("Global vs Per-Category AUROC", fontsize=13, fontweight="bold")
            axes[0].axvline(0.5, color="gray", linestyle=":", lw=1)
            axes[0].legend(loc="lower right")
            axes[0].grid(True, alpha=0.3, axis="x")
            axes[0].set_xlim(0, 1.15)

            # Improvement chart
            colors_imp = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_cmp["improvement"]]
            axes[1].barh(
                df_cmp["category"],
                df_cmp["improvement"],
                color=colors_imp,
                edgecolor="black",
            )
            axes[1].axvline(0, color="black", lw=1)
            axes[1].set_xlabel("AUROC Improvement (Per-Category − Global)")
            axes[1].set_title(
                "Per-Category Improvement over Global Model",
                fontsize=13,
                fontweight="bold",
            )
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
                os.path.join(EVAL_OUTPUT_DIR, "comparison_global_vs_per_category.png"),
                dpi=150,
                bbox_inches="tight",
            )
            logger.info("  → comparison_global_vs_per_category.png")

            # Summary stats
            improved = sum(1 for v in df_cmp["improvement"] if v > 0)
            degraded = sum(1 for v in df_cmp["improvement"] if v < 0)
            avg_improvement = df_cmp["improvement"].mean()
            logger.info(
                "Comparison: %d improved, %d degraded, avg improvement: %+.4f",
                improved,
                degraded,
                avg_improvement,
            )

            # Save comparison CSV
            df_cmp.to_csv(
                os.path.join(EVAL_OUTPUT_DIR, "comparison_global_vs_per_category.csv"),
                index=False,
            )
            logger.info("  → comparison_global_vs_per_category.csv")

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-category autoencoder models.")
    parser.add_argument(
        "--compare-with",
        "-cw",
        type=str,
        default="autoencoder",
        help="Global model to compare against (default: autoencoder). Use 'none' to skip comparison.",
    )
    args = parser.parse_args()

    compare_with = args.compare_with if args.compare_with.lower() != "none" else None
    evaluate_all(compare_global_model=compare_with)


if __name__ == "__main__":
    main()

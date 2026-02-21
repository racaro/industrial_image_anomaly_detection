"""
Unified Evaluation for Anomaly Detection Models
================================================
Supports both the Autoencoder and GAN (Generator) models.

Metrics:
  - Reconstruction error (MSE, MAE, SSIM) per image
  - Error distribution: test/good vs test/anomaly
  - AUROC and Average Precision
  - Per-category metrics
  - Visualization: reconstructions + error maps

Usage:
    python src/evaluate.py --model autoencoder
    python src/evaluate.py --model gan
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path so `src.*` and `models.*` imports work
# regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import BATCH_SIZE, DATASET_PATH, DEVICE, IMG_HEIGHT, IMG_WIDTH, OUTPUTS_DIR, ensure_dataset
from src.dataset import EvalImageDataset, collect_test_images
from src.metrics import compute_ssim_batch
from src.models.autoencoder import Autoencoder
from src.models.gan import Generator


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    "autoencoder": {
        "class": Autoencoder,
        "weights": os.path.join(OUTPUTS_DIR, "autoencoder", "model.pth"),
        "eval_dir": os.path.join(OUTPUTS_DIR, "autoencoder", "evaluation"),
    },
    "gan": {
        "class": Generator,
        "weights": os.path.join(OUTPUTS_DIR, "gan", "generator.pth"),
        "eval_dir": os.path.join(OUTPUTS_DIR, "gan", "evaluation"),
    },
}


def load_model(model_name: str) -> tuple:
    """Load and return (model, output_dir) for the chosen model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[model_name]
    model = info["class"]()

    weights_path = info["weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            f"Train with: python src/models/{model_name}/train.py"
        )

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    eval_dir = info["eval_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    return model, eval_dir


# ──────────────────────────────────────────────
# EVALUATION PIPELINE
# ──────────────────────────────────────────────

def evaluate(model_name: str):
    # --- Load model ---
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'='*60}")

    model, output_dir = load_model(model_name)
    print(f"  Model loaded from:  {MODEL_REGISTRY[model_name]['weights']}")
    print(f"  Results will go to: {output_dir}")

    # --- Collect test images ---
    records = collect_test_images(DATASET_PATH)
    df_records = pd.DataFrame(records)
    print(f"\nTest images found: {len(records)}")
    print(df_records.groupby(["label_name"]).size().to_string())
    print()
    print(df_records.groupby(["category", "label_name"]).size().unstack(fill_value=0).to_string())

    # --- DataLoader ---
    eval_dataset = EvalImageDataset(records, IMG_HEIGHT, IMG_WIDTH)
    num_workers = 0 if os.name == "nt" else 4
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers)

    # --- Compute reconstruction errors ---
    all_mse = []
    all_mae = []
    all_ssim = []
    all_labels = []
    all_indices = []

    print("\nComputing reconstruction errors...")
    with torch.no_grad():
        for imgs, labels, indices in tqdm(eval_loader, desc="  Evaluating", unit="batch"):
            imgs_dev = imgs.to(DEVICE)
            preds = model(imgs_dev)

            mse = ((imgs_dev - preds) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            mae = (torch.abs(imgs_dev - preds)).mean(dim=[1, 2, 3]).cpu().numpy()
            ssim = compute_ssim_batch(imgs_dev, preds).cpu().numpy()

            all_mse.extend(mse.tolist())
            all_mae.extend(mae.tolist())
            all_ssim.extend(ssim.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_indices.extend(indices.numpy().tolist())

    # --- Results DataFrame ---
    df_results = pd.DataFrame({
        "category": [records[i]["category"] for i in all_indices],
        "label": all_labels,
        "label_name": [records[i]["label_name"] for i in all_indices],
        "path": [records[i]["path"] for i in all_indices],
        "mse": all_mse,
        "mae": all_mae,
        "ssim": all_ssim,
    })

    # ── Global metrics ──
    y_true = np.array(all_labels)
    scores_mse = np.array(all_mse)
    scores_ssim = 1 - np.array(all_ssim)

    auroc_mse = roc_auc_score(y_true, scores_mse)
    auroc_ssim = roc_auc_score(y_true, scores_ssim)
    ap_mse = average_precision_score(y_true, scores_mse)
    ap_ssim = average_precision_score(y_true, scores_ssim)

    print(f"\n{'='*60}")
    print(f"  GLOBAL ANOMALY DETECTION METRICS – {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  AUROC  (MSE) :  {auroc_mse:.4f}")
    print(f"  AUROC  (SSIM):  {auroc_ssim:.4f}")
    print(f"  Avg Precision (MSE) :  {ap_mse:.4f}")
    print(f"  Avg Precision (SSIM):  {ap_ssim:.4f}")

    # Error statistics by group
    print(f"\n{'-'*60}")
    print(f"  Reconstruction Error Statistics")
    print(f"{'-'*60}")
    for label_name in ["good", "anomaly"]:
        subset = df_results[df_results["label_name"] == label_name]
        print(f"\n  [{label_name.upper()}] ({len(subset)} images)")
        print(f"    MSE  -> mean: {subset['mse'].mean():.6f}  std: {subset['mse'].std():.6f}  "
              f"min: {subset['mse'].min():.6f}  max: {subset['mse'].max():.6f}")
        print(f"    MAE  -> mean: {subset['mae'].mean():.6f}  std: {subset['mae'].std():.6f}  "
              f"min: {subset['mae'].min():.6f}  max: {subset['mae'].max():.6f}")
        print(f"    SSIM -> mean: {subset['ssim'].mean():.4f}    std: {subset['ssim'].std():.4f}    "
              f"min: {subset['ssim'].min():.4f}    max: {subset['ssim'].max():.4f}")

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, scores_mse)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (scores_mse >= best_threshold).astype(int)

    print(f"\n  Optimal MSE threshold (Youden's J): {best_threshold:.6f}")
    print(f"\n  Classification Report (optimal threshold):")
    print(classification_report(y_true, y_pred, target_names=["good", "anomaly"]))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    {cm}")

    # ── Per-category metrics ──
    print(f"\n{'='*60}")
    print(f"  AUROC PER CATEGORY – {model_name.upper()}")
    print(f"{'='*60}")
    cat_metrics = []
    for cat in sorted(df_results["category"].unique()):
        cat_df = df_results[df_results["category"] == cat]
        if cat_df["label"].nunique() < 2:
            print(f"  {cat:15s}  Warning: Only one class present, cannot compute AUROC")
            continue
        cat_auroc = roc_auc_score(cat_df["label"], cat_df["mse"])
        cat_ap = average_precision_score(cat_df["label"], cat_df["mse"])
        n_good = (cat_df["label"] == 0).sum()
        n_anom = (cat_df["label"] == 1).sum()
        cat_metrics.append({
            "Category": cat, "N_Good": n_good, "N_Anomaly": n_anom,
            "AUROC": cat_auroc, "Avg_Precision": cat_ap,
        })
        print(f"  {cat:15s}  AUROC: {cat_auroc:.4f}  AP: {cat_ap:.4f}  "
              f"(good={n_good}, anomaly={n_anom})")

    df_cat_metrics = pd.DataFrame(cat_metrics)

    # ── Visualizations ──
    print("\nGenerating visualizations...")

    # 1. ROC + Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(fpr, tpr, "b-", lw=2, label=f"MSE (AUROC={auroc_mse:.3f})")
    fpr_s, tpr_s, _ = roc_curve(y_true, scores_ssim)
    axes[0].plot(fpr_s, tpr_s, "r--", lw=2, label=f"1-SSIM (AUROC={auroc_ssim:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].scatter([fpr[best_idx]], [tpr[best_idx]], c="green", s=100, zorder=5,
                    label=f"Optimal threshold={best_threshold:.4f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, scores_mse)
    prec_s, rec_s, _ = precision_recall_curve(y_true, scores_ssim)
    axes[1].plot(rec, prec, "b-", lw=2, label=f"MSE (AP={ap_mse:.3f})")
    axes[1].plot(rec_s, prec_s, "r--", lw=2, label=f"1-SSIM (AP={ap_ssim:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Anomaly Detection Performance – {model_name.upper()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    print(f"  -> roc_pr_curves.png")

    # 2. MSE error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    good_mse = df_results[df_results["label"] == 0]["mse"]
    anom_mse = df_results[df_results["label"] == 1]["mse"]
    ax.hist(good_mse, bins=50, alpha=0.6, label=f"Good (n={len(good_mse)})", color="green")
    ax.hist(anom_mse, bins=50, alpha=0.6, label=f"Anomaly (n={len(anom_mse)})", color="red")
    ax.axvline(best_threshold, color="black", linestyle="--", lw=2,
               label=f"Threshold={best_threshold:.4f}")
    ax.set_xlabel("MSE (Reconstruction Error)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Reconstruction Error Distribution – {model_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"  -> error_distribution.png")

    # 3. AUROC per category (bar chart)
    if len(df_cat_metrics) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ["green" if v >= 0.8 else "orange" if v >= 0.6 else "red"
                  for v in df_cat_metrics["AUROC"]]
        bars = ax.bar(df_cat_metrics["Category"], df_cat_metrics["AUROC"],
                      color=colors, edgecolor="black")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="Random (0.5)")
        ax.axhline(auroc_mse, color="blue", linestyle="--", lw=2,
                    label=f"Global AUROC={auroc_mse:.3f}")
        for bar, val in zip(bars, df_cat_metrics["AUROC"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel("AUROC")
        ax.set_title(f"AUROC per Category – {model_name.upper()}")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auroc_per_category.png"),
                    dpi=150, bbox_inches="tight")
        print(f"  -> auroc_per_category.png")

    # 4. Reconstruction samples + error maps
    print("\nGenerating reconstruction samples...")
    n_samples = 8
    sample_good = df_results[df_results["label"] == 0].sample(
        n=min(n_samples, len(good_mse)), random_state=42
    )
    sample_anom = df_results[df_results["label"] == 1].nlargest(
        min(n_samples, len(anom_mse)), "mse"
    )

    tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    for group_name, sample_df in [("good", sample_good), ("anomaly", sample_anom)]:
        n = len(sample_df)
        fig, axes_grid = plt.subplots(3, n, figsize=(3 * n, 9))
        if n == 1:
            axes_grid = axes_grid.reshape(3, 1)

        for i, (_, row) in enumerate(sample_df.iterrows()):
            img = Image.open(row["path"]).convert("RGB")
            img_tensor = tf(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                recon = model(img_tensor)

            orig_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            recon_np = recon.squeeze().cpu().permute(1, 2, 0).numpy()
            error_map = np.mean((orig_np - recon_np) ** 2, axis=2)

            axes_grid[0, i].imshow(orig_np)
            axes_grid[0, i].set_title(f"{row['category']}\n({group_name})", fontsize=9)
            axes_grid[0, i].axis("off")

            axes_grid[1, i].imshow(recon_np)
            axes_grid[1, i].set_title(f"MSE: {row['mse']:.5f}", fontsize=9)
            axes_grid[1, i].axis("off")

            axes_grid[2, i].imshow(error_map, cmap="hot", vmin=0, vmax=error_map.max())
            axes_grid[2, i].set_title("Error Map", fontsize=9)
            axes_grid[2, i].axis("off")

        axes_grid[0, 0].set_ylabel("Original", fontsize=11, fontweight="bold")
        axes_grid[1, 0].set_ylabel("Reconstruction", fontsize=11, fontweight="bold")
        axes_grid[2, 0].set_ylabel("Error Map", fontsize=11, fontweight="bold")

        plt.suptitle(f"Reconstructions – {group_name.upper()} ({model_name.upper()})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"reconstructions_{group_name}.png"),
                    dpi=150, bbox_inches="tight")
        print(f"  -> reconstructions_{group_name}.png")

    # Save CSV
    df_results.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    if len(df_cat_metrics) > 0:
        df_cat_metrics.to_csv(os.path.join(output_dir, "metrics_per_category.csv"), index=False)
    print(f"\n  -> evaluation_results.csv")
    print(f"  -> metrics_per_category.csv")

    # Save JSON
    good_subset = df_results[df_results["label_name"] == "good"]
    anom_subset = df_results[df_results["label_name"] == "anomaly"]

    results_json = {
        "model": model_name,
        "global_metrics": {
            "auroc_mse": round(float(auroc_mse), 4),
            "auroc_ssim": round(float(auroc_ssim), 4),
            "avg_precision_mse": round(float(ap_mse), 4),
            "avg_precision_ssim": round(float(ap_ssim), 4),
            "optimal_threshold_mse": round(float(best_threshold), 6),
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "error_statistics": {},
        "per_category": [],
    }

    for group_label, grp_df in [("good", good_subset), ("anomaly", anom_subset)]:
        results_json["error_statistics"][group_label] = {
            "count": int(len(grp_df)),
            "mse": {
                "mean": round(float(grp_df["mse"].mean()), 6),
                "std": round(float(grp_df["mse"].std()), 6),
                "min": round(float(grp_df["mse"].min()), 6),
                "max": round(float(grp_df["mse"].max()), 6),
            },
            "mae": {
                "mean": round(float(grp_df["mae"].mean()), 6),
                "std": round(float(grp_df["mae"].std()), 6),
                "min": round(float(grp_df["mae"].min()), 6),
                "max": round(float(grp_df["mae"].max()), 6),
            },
            "ssim": {
                "mean": round(float(grp_df["ssim"].mean()), 4),
                "std": round(float(grp_df["ssim"].std()), 4),
                "min": round(float(grp_df["ssim"].min()), 4),
                "max": round(float(grp_df["ssim"].max()), 4),
            },
        }

    if len(df_cat_metrics) > 0:
        results_json["per_category"] = [
            {
                "category": row["Category"],
                "n_good": int(row["N_Good"]),
                "n_anomaly": int(row["N_Anomaly"]),
                "auroc": round(float(row["AUROC"]), 4),
                "avg_precision": round(float(row["Avg_Precision"]), 4),
            }
            for _, row in df_cat_metrics.iterrows()
        ]

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"  -> evaluation_results.json")

    print(f"\nAll results saved to: {output_dir}")
    return df_results, df_cat_metrics


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models.")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_REGISTRY.keys()),
        default="autoencoder",
        help="Which model to evaluate (default: autoencoder)",
    )
    args = parser.parse_args()
    evaluate(args.model)


if __name__ == "__main__":
    main()

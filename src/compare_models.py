"""
Model comparison: Autoencoder vs GAN vs Diffusion.

Generates publication-ready comparison charts including global metrics,
per-category AUROC heatmap, error distributions, radar chart,
confusion matrices, and summary ranking table.
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, OUTPUTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_NAMES = ["autoencoder", "autoencoder_v2", "gan", "diffusion"]
MODEL_COLORS = {
    "autoencoder": "#2196F3",  # Blue
    "autoencoder_v2": "#9C27B0",  # Purple
    "gan": "#FF9800",  # Orange
    "diffusion": "#4CAF50",  # Green
}
MODEL_LABELS = {
    "autoencoder": "Autoencoder V1",
    "autoencoder_v2": "Autoencoder V2",
    "gan": "GAN",
    "diffusion": "Diffusion (DDPM)",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def load_evaluation_results() -> dict[str, dict]:
    """Load evaluation_results.json for all available models."""
    results = {}
    for model_name in MODEL_NAMES:
        json_path = os.path.join(OUTPUTS_DIR, model_name, "evaluation", "evaluation_results.json")
        if os.path.isfile(json_path):
            with open(json_path, encoding="utf-8") as f:
                results[model_name] = json.load(f)
            logger.info("Loaded results for: %s", model_name)
        else:
            logger.warning("No evaluation results found for: %s (expected at %s)", model_name, json_path)
    return results


def load_evaluation_csvs() -> dict[str, pd.DataFrame]:
    """Load evaluation_results.csv for all available models."""
    dfs = {}
    for model_name in MODEL_NAMES:
        csv_path = os.path.join(OUTPUTS_DIR, model_name, "evaluation", "evaluation_results.csv")
        if os.path.isfile(csv_path):
            dfs[model_name] = pd.read_csv(csv_path)
            logger.info("Loaded CSV for: %s (%d rows)", model_name, len(dfs[model_name]))
    return dfs


def plot_global_metrics(results: dict[str, dict], save_dir: str) -> None:
    """Bar chart comparing AUROC and AP across models (all scoring methods)."""
    models = list(results.keys())
    metrics = [
        "auroc_mse",
        "auroc_ssim",
        "auroc_perceptual",
        "auroc_combined",
        "avg_precision_mse",
        "avg_precision_ssim",
        "avg_precision_perceptual",
        "avg_precision_combined",
    ]
    metric_labels = [
        "AUROC\n(MSE)",
        "AUROC\n(SSIM)",
        "AUROC\n(Percept.)",
        "AUROC\n(Combined)",
        "AP\n(MSE)",
        "AP\n(SSIM)",
        "AP\n(Percept.)",
        "AP\n(Combined)",
    ]

    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(models))

    for i, model in enumerate(models):
        values = [results[model]["global_metrics"].get(m, 0) for m in metrics]
        bars = ax.bar(
            x + offsets[i],
            values,
            width * 0.9,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        # Value labels
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Global Anomaly Detection Metrics — Model Comparison (All Scoring Methods)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.6, label="Random baseline (0.5)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_global_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def plot_category_heatmap(results: dict[str, dict], save_dir: str) -> None:
    """Heatmap of AUROC per category per model."""
    models = list(results.keys())

    # Collect all categories
    all_categories = set()
    for model in models:
        for cat_info in results[model].get("per_category", []):
            all_categories.add(cat_info["category"])
    categories = sorted(all_categories)

    # Build matrix
    data = np.full((len(categories), len(models)), np.nan)
    for j, model in enumerate(models):
        cat_dict = {c["category"]: c["auroc"] for c in results[model].get("per_category", [])}
        for i, cat in enumerate(categories):
            data[i, j] = cat_dict.get(cat, np.nan)

    fig, ax = plt.subplots(figsize=(8, max(10, len(categories) * 0.45)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(models)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    # Highlight best per category
    for i in range(len(categories)):
        row = data[i]
        if not np.all(np.isnan(row)):
            best_j = np.nanargmax(row)
            rect = plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="gold", linewidth=2.5)
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    ax.set_title(
        "AUROC per Category — Model Comparison\n(Gold border = best model)", fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_auroc_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def plot_error_distributions(dfs: dict[str, pd.DataFrame], save_dir: str) -> None:
    """Box plots of MSE distributions (good vs anomaly) per model."""
    models = list(dfs.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        df = dfs[model]
        good = df[df["label"] == 0]["mse"]
        anom = df[df["label"] == 1]["mse"]

        bp = ax.boxplot(
            [good, anom],
            tick_labels=["Good", "Anomaly"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#81C784")
        bp["boxes"][1].set_facecolor("#E57373")

        ax.set_title(
            MODEL_LABELS.get(model, model), fontsize=13, fontweight="bold", color=MODEL_COLORS.get(model, "black")
        )
        ax.set_ylabel("MSE (Reconstruction Error)")
        ax.grid(True, alpha=0.3)

        # Add count annotations
        ax.text(1, good.max() * 1.02, f"n={len(good)}", ha="center", fontsize=9, color="gray")
        ax.text(2, anom.max() * 1.02, f"n={len(anom)}", ha="center", fontsize=9, color="gray")

    plt.suptitle("Reconstruction Error Distribution — Good vs Anomaly", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_error_distributions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def plot_radar_chart(results: dict[str, dict], save_dir: str) -> None:
    """Radar chart comparing models across all categories."""
    models = list(results.keys())

    # Get shared categories
    all_categories = set()
    for model in models:
        for c in results[model].get("per_category", []):
            all_categories.add(c["category"])
    categories = sorted(all_categories)

    if len(categories) < 3:
        logger.warning("Not enough categories for radar chart (need >= 3)")
        return

    # Build values
    model_values = {}
    for model in models:
        cat_dict = {c["category"]: c["auroc"] for c in results[model].get("per_category", [])}
        model_values[model] = [cat_dict.get(cat, 0.0) for cat in categories]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model in models:
        values = model_values[model] + model_values[model][:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            markersize=4,
        )
        ax.fill(angles, values, alpha=0.08, color=MODEL_COLORS.get(model, "gray"))

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.set_title("AUROC per Category — Radar View", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_radar_chart.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def plot_category_bars(results: dict[str, dict], save_dir: str) -> None:
    """Grouped bar chart of AUROC per category."""
    models = list(results.keys())

    all_categories = set()
    for model in models:
        for c in results[model].get("per_category", []):
            all_categories.add(c["category"])
    categories = sorted(all_categories)

    fig, ax = plt.subplots(figsize=(max(16, len(categories) * 0.8), 7))
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        cat_dict = {c["category"]: c["auroc"] for c in results[model].get("per_category", [])}
        values = [cat_dict.get(cat, 0.0) for cat in categories]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width * 0.9,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.6)
    ax.set_title("AUROC per Category — All Models", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_auroc_per_category.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def plot_confusion_matrices(results: dict[str, dict], save_dir: str) -> None:
    """Side-by-side confusion matrices."""
    models = list(results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        cm_data = results[model]["confusion_matrix"]
        cm = np.array(
            [
                [cm_data["true_negatives"], cm_data["false_positives"]],
                [cm_data["false_negatives"], cm_data["true_positives"]],
            ]
        )

        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Good", "Pred Anomaly"])
        ax.set_yticklabels(["True Good", "True Anomaly"])
        ax.set_title(
            MODEL_LABELS.get(model, model), fontsize=13, fontweight="bold", color=MODEL_COLORS.get(model, "black")
        )

        for i in range(2):
            for j in range(2):
                total = cm.sum()
                pct = cm[i, j] / total * 100
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

    plt.suptitle("Confusion Matrices — Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_confusion_matrices.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def generate_summary_table(results: dict[str, dict], save_dir: str) -> pd.DataFrame:
    """Generate and save a summary comparison table."""
    rows = []
    for model in results:
        gm = results[model]["global_metrics"]
        es = results[model]["error_statistics"]
        cats = results[model].get("per_category", [])

        cat_aurocs = [c["auroc"] for c in cats]
        best_cats = [c["category"] for c in cats if c["auroc"] >= 0.9]
        worst_cats = [c["category"] for c in cats if c["auroc"] < 0.5]

        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "AUROC (MSE)": gm["auroc_mse"],
                "AUROC (SSIM)": gm["auroc_ssim"],
                "AUROC (Percept.)": gm.get("auroc_perceptual", 0),
                "AUROC (Combined)": gm.get("auroc_combined", 0),
                "AP (MSE)": gm["avg_precision_mse"],
                "AP (Combined)": gm.get("avg_precision_combined", 0),
                "Mean AUROC/cat": np.mean(cat_aurocs) if cat_aurocs else 0,
                "Median AUROC/cat": np.median(cat_aurocs) if cat_aurocs else 0,
                "Categories >= 0.9": len(best_cats),
                "Categories < 0.5": len(worst_cats),
                "MSE Good (mean)": es["good"]["mse"]["mean"],
                "MSE Anomaly (mean)": es["anomaly"]["mse"]["mean"],
                "MSE Separation": es["anomaly"]["mse"]["mean"] - es["good"]["mse"]["mean"],
            }
        )

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(save_dir, "comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info("  → %s", csv_path)

    # Create a nice figure table
    fig, ax = plt.subplots(figsize=(20, 3 + len(rows) * 0.8))
    ax.axis("off")

    # Format for display
    display_cols = [
        "Model",
        "AUROC (MSE)",
        "AUROC (SSIM)",
        "AUROC (Percept.)",
        "AUROC (Combined)",
        "AP (MSE)",
        "AP (Combined)",
        "Mean AUROC/cat",
        "Categories >= 0.9",
        "Categories < 0.5",
        "MSE Separation",
    ]
    df_display = df[display_cols].copy()
    for col in display_cols[1:]:
        if col not in ["Categories >= 0.9", "Categories < 0.5"]:
            df_display[col] = df_display[col].map(lambda v: f"{v:.4f}")
        else:
            df_display[col] = df_display[col].astype(int)

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(display_cols)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best values per column
    for j, col in enumerate(display_cols[1:], start=1):
        values = df[col].values
        if col == "Categories < 0.5":
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        table[best_idx + 1, j].set_facecolor("#C8E6C9")

    ax.set_title("Model Comparison Summary", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_summary_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)

    return df


def plot_ssim_distributions(dfs: dict[str, pd.DataFrame], save_dir: str) -> None:
    """Violin plots of SSIM distributions per model and label."""
    models = list(dfs.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        df = dfs[model]
        good_ssim = df[df["label"] == 0]["ssim"].values
        anom_ssim = df[df["label"] == 1]["ssim"].values

        parts = ax.violinplot([good_ssim, anom_ssim], positions=[1, 2], showmeans=True, showmedians=True)

        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(["#81C784", "#E57373"][i])
            body.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("blue")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Good", "Anomaly"])
        ax.set_title(
            MODEL_LABELS.get(model, model), fontsize=13, fontweight="bold", color=MODEL_COLORS.get(model, "black")
        )
        ax.set_ylabel("SSIM")
        ax.grid(True, alpha=0.3)

    plt.suptitle("SSIM Distribution — Good vs Anomaly", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_ssim_distributions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


def determine_best_model(results: dict[str, dict]) -> str:
    """Determine the best model based on multiple criteria."""
    logger.info("=" * 60)
    logger.info("  MODEL RANKING")
    logger.info("=" * 60)

    scores = {model: 0.0 for model in results}

    # Weight different metrics
    criteria = [
        ("auroc_mse", 2.0, True),
        ("auroc_ssim", 1.5, True),
        ("auroc_perceptual", 2.5, True),
        ("auroc_combined", 3.0, True),
        ("avg_precision_mse", 1.5, True),
        ("avg_precision_combined", 2.5, True),
    ]

    for metric, weight, higher_better in criteria:
        values = {m: results[m]["global_metrics"].get(metric, 0) for m in results}
        ranked = sorted(values.items(), key=lambda x: x[1], reverse=higher_better)
        for rank, (model, val) in enumerate(ranked):
            points = (len(ranked) - rank) * weight
            scores[model] += points
            logger.info("  %s | %s = %.4f → +%.1f pts", model, metric, val, points)

    # Mean per-category AUROC (important for robustness)
    for model in results:
        cat_aurocs = [c["auroc"] for c in results[model].get("per_category", [])]
        mean_cat = np.mean(cat_aurocs) if cat_aurocs else 0
        scores[model] += mean_cat * 5.0  # weight 5
        logger.info("  %s | mean_cat_auroc = %.4f → +%.1f pts", model, mean_cat, mean_cat * 5.0)

    logger.info("-" * 60)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, score) in enumerate(ranking, 1):
        marker = " ★" if rank == 1 else ""
        logger.info(
            "  #%d  %s  (%.1f pts)%s",
            rank,
            MODEL_LABELS.get(model, model),
            score,
            marker,
        )

    best_model = ranking[0][0]
    logger.info("=" * 60)
    logger.info("  BEST MODEL: %s", MODEL_LABELS.get(best_model, best_model))
    logger.info("=" * 60)
    return best_model


def main() -> None:
    """Generate all comparison charts and determine the best model."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  MODEL COMPARISON — All Models")
    logger.info("=" * 60)

    # Load results
    results = load_evaluation_results()
    dfs = load_evaluation_csvs()

    if len(results) < 2:
        logger.error("Need at least 2 models with evaluation results. Found: %s", list(results.keys()))
        logger.error("Run evaluations first: python src/evaluate.py --model <name>")
        sys.exit(1)

    logger.info("Models with results: %s", list(results.keys()))
    logger.info("Generating comparison charts...")

    # Generate all charts
    plot_global_metrics(results, FIGURES_DIR)
    plot_category_heatmap(results, FIGURES_DIR)
    plot_category_bars(results, FIGURES_DIR)
    plot_radar_chart(results, FIGURES_DIR)
    plot_confusion_matrices(results, FIGURES_DIR)
    plot_ssim_distributions(dfs, FIGURES_DIR)
    plot_error_distributions(dfs, FIGURES_DIR)
    summary_df = generate_summary_table(results, FIGURES_DIR)

    # Print summary
    logger.info("\n%s", summary_df.to_string(index=False))

    # Determine best model
    best = determine_best_model(results)

    logger.info("\nAll comparison charts saved to: %s", FIGURES_DIR)
    logger.info("Done.")

    return results, summary_df, best


if __name__ == "__main__":
    main()

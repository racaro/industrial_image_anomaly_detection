"""
Final comparison: Global AE vs Per-Category AE vs PatchCore.

Generates side-by-side comparison charts and a summary table
across all three anomaly detection approaches.
"""

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

from src.config import FIGURES_DIR, OUTPUTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

RESULTS_MAP = {
    "Global AE V1": os.path.join(OUTPUTS_DIR, "autoencoder", "evaluation", "evaluation_results.json"),
    "Per-Category AE": os.path.join(OUTPUTS_DIR, "autoencoder_per_category", "evaluation", "evaluation_results.json"),
    "PatchCore": os.path.join(OUTPUTS_DIR, "patchcore", "evaluation", "evaluation_results.json"),
}


def load_results() -> dict[str, dict]:
    """Load evaluation results for all available approaches."""
    results = {}
    for name, path in RESULTS_MAP.items():
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                results[name] = json.load(f)
            logger.info("Loaded: %s from %s", name, path)
        else:
            logger.warning("Results not found for %s at %s", name, path)
    return results


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a per-category comparison table across all approaches.

    Returns:
        DataFrame with columns: Category, <approach>_auroc, ...
    """
    # Collect per-category AUROC from each approach
    cat_data: dict[str, dict[str, float | None]] = {}

    for approach_name, data in results.items():
        per_cat = data.get("per_category", [])
        for entry in per_cat:
            cat = entry.get("category", "")
            # Different result formats
            auroc = entry.get("auroc_combined") or entry.get("auroc")
            if cat:
                cat_data.setdefault(cat, {})[approach_name] = auroc

    # Build DataFrame
    rows = []
    for cat in sorted(cat_data.keys()):
        row = {"Category": cat}
        for approach_name in results:
            row[approach_name] = cat_data[cat].get(approach_name)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_comparison_charts(
    df: pd.DataFrame,
    results: dict[str, dict],
) -> None:
    """Generate comprehensive comparison charts."""

    os.makedirs(FIGURES_DIR, exist_ok=True)

    approach_names = [col for col in df.columns if col != "Category"]
    approach_colors = {
        "Global AE V1": "#e74c3c",
        "Per-Category AE": "#2ecc71",
        "PatchCore": "#3498db",
    }

    # Per-category AUROC grouped bar chart
    valid_df = df.dropna(subset=approach_names, how="all")
    valid_df = valid_df.sort_values(
        approach_names[-1] if approach_names else "Category",
        ascending=True,
        na_position="first",
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    n_approaches = len(approach_names)
    bar_height = 0.8 / n_approaches
    y = np.arange(len(valid_df))

    for i, approach in enumerate(approach_names):
        values = valid_df[approach].fillna(0).values
        color = approach_colors.get(approach, f"C{i}")
        offset = (i - n_approaches / 2 + 0.5) * bar_height
        ax.barh(
            y + offset,
            values,
            bar_height,
            label=approach,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(valid_df["Category"], fontsize=9)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_title(
        "Anomaly Detection: AUROC by Category — All Approaches",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(0.5, color="gray", linestyle=":", lw=1, label="Random (0.5)")
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "comparison_all_approaches_per_category.png"),
        dpi=150,
        bbox_inches="tight",
    )
    logger.info("  → comparison_all_approaches_per_category.png")

    # Summary bar chart (mean/median/weighted AUROC)
    summary_data = []
    for name, data in results.items():
        entry = {"Approach": name}

        if "global_metrics" in data:
            entry["Global AUROC"] = data["global_metrics"].get("auroc_combined")
        else:
            entry["Global AUROC"] = data.get("global_auroc_aggregated")

        per_cat = data.get("per_category", [])
        aurocs = [
            e.get("auroc_combined") or e.get("auroc")
            for e in per_cat
            if (e.get("auroc_combined") or e.get("auroc")) is not None
        ]
        if aurocs:
            entry["Mean AUROC"] = np.mean(aurocs)
            entry["Median AUROC"] = np.median(aurocs)
            entry["Best Category"] = max(aurocs)
            entry["Worst Category"] = min(aurocs)
            entry["Num ≥ 0.8"] = sum(1 for a in aurocs if a >= 0.8)
            entry["Num ≥ 0.9"] = sum(1 for a in aurocs if a >= 0.9)
        summary_data.append(entry)

    df_summary = pd.DataFrame(summary_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Mean/Median AUROC comparison
    metrics = ["Mean AUROC", "Median AUROC"]
    x = np.arange(len(df_summary))
    width = 0.3

    for i, metric in enumerate(metrics):
        values = df_summary[metric].fillna(0).values
        color = ["#e74c3c", "#2ecc71", "#3498db"][: len(values)]
        axes[0].bar(
            x + i * width - width / 2,
            values,
            width,
            label=metric,
            color=color if i == 0 else [c + "80" for c in color],
            edgecolor="black",
        )
        for j, v in enumerate(values):
            if v > 0:
                axes[0].text(
                    j + i * width - width / 2,
                    v + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_summary["Approach"], fontsize=10)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Mean & Median AUROC by Approach", fontsize=13, fontweight="bold")
    axes[0].axhline(0.5, color="gray", linestyle=":", lw=1)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Categories ≥ 0.8 and ≥ 0.9
    bar_width = 0.35
    categories_08 = df_summary["Num ≥ 0.8"].fillna(0).values
    categories_09 = df_summary["Num ≥ 0.9"].fillna(0).values
    colors = ["#e74c3c", "#2ecc71", "#3498db"][: len(df_summary)]

    axes[1].bar(
        x - bar_width / 2, categories_08, bar_width, label="AUROC ≥ 0.8", color=colors, alpha=0.7, edgecolor="black"
    )
    axes[1].bar(
        x + bar_width / 2, categories_09, bar_width, label="AUROC ≥ 0.9", color=colors, alpha=1.0, edgecolor="black"
    )

    for j in range(len(x)):
        axes[1].text(
            x[j] - bar_width / 2, categories_08[j] + 0.3, f"{int(categories_08[j])}", ha="center", fontweight="bold"
        )
        axes[1].text(
            x[j] + bar_width / 2, categories_09[j] + 0.3, f"{int(categories_09[j])}", ha="center", fontweight="bold"
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_summary["Approach"], fontsize=10)
    axes[1].set_ylabel("Number of Categories")
    axes[1].set_title("Categories Achieving High AUROC", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Anomaly Detection — Three Approaches Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "comparison_all_approaches_summary.png"),
        dpi=150,
        bbox_inches="tight",
    )
    logger.info("  → comparison_all_approaches_summary.png")

    # Improvement heatmap
    if len(approach_names) >= 2:
        fig, ax = plt.subplots(figsize=(10, 12))

        # Use the last approach (PatchCore) as reference for improvement over first (Global)
        if "Global AE V1" in approach_names and len(approach_names) >= 2:
            heatmap_data = valid_df.set_index("Category")[approach_names].copy()

            im = ax.imshow(
                heatmap_data.values,
                cmap="RdYlGn",
                aspect="auto",
                vmin=0,
                vmax=1,
            )
            ax.set_xticks(range(len(approach_names)))
            ax.set_xticklabels(approach_names, fontsize=10, rotation=15)
            ax.set_yticks(range(len(heatmap_data)))
            ax.set_yticklabels(heatmap_data.index, fontsize=9)

            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    val = heatmap_data.values[i, j]
                    if not np.isnan(val):
                        color = "white" if val < 0.4 or val > 0.85 else "black"
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color
                        )

            plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
            ax.set_title(
                "AUROC Heatmap — All Approaches × Categories",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(FIGURES_DIR, "comparison_heatmap.png"),
                dpi=150,
                bbox_inches="tight",
            )
            logger.info("  → comparison_heatmap.png")

    plt.close("all")


def main():
    logger.info("=" * 60)
    logger.info("  FINAL COMPARISON: ALL APPROACHES")
    logger.info("=" * 60)

    results = load_results()
    if not results:
        logger.error("No results found. Run evaluations first.")
        return

    df = build_comparison_table(results)

    # Print summary table
    logger.info("\n%s", df.to_markdown(index=False, floatfmt=".4f"))

    # Per-approach summary
    for name, data in results.items():
        per_cat = data.get("per_category", [])
        aurocs = [
            e.get("auroc_combined") or e.get("auroc")
            for e in per_cat
            if (e.get("auroc_combined") or e.get("auroc")) is not None
        ]
        if aurocs:
            logger.info(
                "%-20s  Mean: %.4f  Median: %.4f  Best: %.4f  Worst: %.4f  (≥0.8: %d, ≥0.9: %d)",
                name,
                np.mean(aurocs),
                np.median(aurocs),
                max(aurocs),
                min(aurocs),
                sum(1 for a in aurocs if a >= 0.8),
                sum(1 for a in aurocs if a >= 0.9),
            )

    generate_comparison_charts(df, results)

    # Save comparison table
    comparison_path = os.path.join(FIGURES_DIR, "comparison_all_approaches.csv")
    df.to_csv(comparison_path, index=False)
    logger.info("Comparison table saved → %s", comparison_path)

    # Save summary JSON
    summary = {
        "approaches": list(results.keys()),
        "comparison": df.to_dict(orient="records"),
    }

    for name, data in results.items():
        per_cat = data.get("per_category", [])
        aurocs = [
            e.get("auroc_combined") or e.get("auroc")
            for e in per_cat
            if (e.get("auroc_combined") or e.get("auroc")) is not None
        ]
        if aurocs:
            summary[name] = {
                "mean_auroc": round(float(np.mean(aurocs)), 4),
                "median_auroc": round(float(np.median(aurocs)), 4),
                "best_auroc": round(float(max(aurocs)), 4),
                "worst_auroc": round(float(min(aurocs)), 4),
                "num_ge_08": sum(1 for a in aurocs if a >= 0.8),
                "num_ge_09": sum(1 for a in aurocs if a >= 0.9),
            }

    json_path = os.path.join(FIGURES_DIR, "comparison_all_approaches.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary JSON saved → %s", json_path)


if __name__ == "__main__":
    main()

# Results & Analysis

This document traces the evolution of the anomaly detection pipeline — from an initial hypothesis through iterative improvements — showing how each design decision impacted performance across 27 industrial inspection categories.

## Environment & Dataset

| Detail | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, Ada Lovelace) |
| **PyTorch** | 2.6.0+cu124 |
| **Dataset** | MVTec AD (15 categories) + VisA (12 categories) = 27 total |
| **Split** | 12,050 training images / 3,168 test images (1,667 good + 1,501 anomaly) |
| **Image size** | 256 × 256 RGB |

---

## Phase 1 — Starting Point: Reconstruction-Based Models

### Hypothesis

> *"If a model is trained to reconstruct only normal images, it will fail to accurately reconstruct anomalies — producing higher reconstruction error that can be used as an anomaly score."*

Four reconstruction-based architectures were trained as **single global models** on all 12,050 training images simultaneously:

| Model | Architecture | Bottleneck | Epochs | Key Design Choice |
|---|---|---|---|---|
| Autoencoder V1 | Conv encoder–decoder | 65K dims | 50 | Wide bottleneck, MSE loss |
| Autoencoder V2 | Conv encoder–decoder | 8K dims (24:1 compression) | 50 | Tight bottleneck, augmentation |
| GAN | Generator + Discriminator | — | 50 | Adversarial reconstruction |
| Diffusion (DDPM) | U-Net denoiser | — | 40 | Single-step denoising at t=250 |

**Scoring**: Each image receives four anomaly scores — MSE, 1−SSIM, Perceptual (LPIPS), and a Combined score (0.3×MSE + 0.3×SSIM + 0.4×Perceptual). Evaluation is done via AUROC and Average Precision (AP).

### Training Convergence

All models converged normally during training:

| Model | Initial Loss | Final Loss | Reduction |
|---|---|---|---|
| Autoencoder V1 | 0.0096 | 0.00036 | 96% |
| Autoencoder V2 | 0.1853 | 0.0510 | 72% |
| GAN | G=4.86, D=0.69 | G=1.56, D=0.41 | G: 68% |
| Diffusion | 0.1975 | 0.0162 | 92% |

### Results: AUROC ≈ 0.5

Despite successful convergence, all models achieved near-random anomaly detection performance:

| Model | AUROC (Combined) | AUROC (MSE) | AP (Combined) | Rank |
|---|---|---|---|---|
| Autoencoder V1 | 0.5339 | 0.5397 | 0.5290 | #1 |
| Autoencoder V2 | 0.5245 | 0.5245 | 0.5243 | #2 |
| GAN | 0.5148 | 0.5351 | 0.4874 | #3 |
| Diffusion (DDPM) | 0.4835 | 0.4960 | 0.4637 | #4 |

### Root Cause Analysis

**Why did models converge but fail to detect anomalies?**

1. **Single model across 27 heterogeneous categories** — Training one model on products as diverse as screws, PCBs, leather, and candles forces it to learn a very general representation. An anomaly in one category (scratch on metal) may resemble a normal pattern in another (texture on wood).

2. **Negligible MSE separation** — The gap between good and anomaly reconstruction errors is extremely small:

   | Model | MSE Good (mean) | MSE Anomaly (mean) | Gap |
   |---|---|---|---|
   | AE V1 | 0.000330 | 0.000367 | +0.000037 |
   | AE V2 | 0.001868 | 0.001979 | +0.000111 |
   | GAN | 0.002447 | 0.002585 | +0.000138 |
   | Diffusion | 0.057427 | 0.057255 | −0.000172 |

3. **Category-level cancellation** — Some categories achieved excellent AUROC (screw: 1.00, cashew: 0.94) while others scored below random (transistor: 0.19, leather: 0.27). When aggregated with a single global threshold, these cancel out toward 0.5.

4. **Diffusion inversion** — The diffusion model actually reconstructed anomalies *better* than normal images (negative MSE gap), because single-step denoising at t=250 is too coarse to preserve fine details.

### Key Insight

> The models **did learn** — some individual categories reached AUROC > 0.90. The problem was **architectural**: using one model for all 27 categories eliminates the ability to set category-specific decision boundaries.

---

## Phase 2 — Improvement: Per-Category Autoencoder

### Strategy

Instead of training one global model, train **27 independent Autoencoder V1 models** — one per category — each using only 30 epochs on that category's normal images. This allows each model to specialize in a single product type.

### Results: Mean AUROC 0.63 → 0.70

| Metric | Global AE V1 | Per-Category AE | Change |
|---|---|---|---|
| Mean AUROC (per-category) | 0.6274 | **0.7012** | +0.0738 |
| Median AUROC (per-category) | — | **0.7389** | — |

Per-category specialization improved the mean by **+12%** relative, with the median at 0.74 indicating most categories benefited. However, the results were mixed:

**Winners** (categories with large gains):

| Category | Global AE V1 | Per-Category AE | Δ |
|---|---|---|---|
| grid | 0.4048 | 0.8810 | +0.4762 |
| candle | 0.4112 | 0.9240 | +0.5128 |
| leather | 0.2714 | 0.5526 | +0.2812 |
| cable | 0.6773 | 0.8448 | +0.1675 |
| capsule | 0.5312 | 0.8261 | +0.2949 |

**Losers** (categories that degraded):

| Category | Global AE V1 | Per-Category AE | Δ |
|---|---|---|---|
| screw | 1.0000 | 0.2888 | −0.7112 |
| tile | 0.9259 | 0.5202 | −0.4057 |
| fryum | 0.9076 | 0.7156 | −0.1920 |
| cashew | 0.9434 | 0.9726 | +0.0292 |

### Diagnosis

Per-category training helped categories with distinctive texture patterns (grid, candle), but hurt categories where the global model had already found good separation (screw, tile). The fundamental limitation of reconstruction-based methods remained: they depend on the model's inability to reconstruct defects, which is unreliable for subtle anomalies.

---

## Phase 3 — Paradigm Shift: PatchCore (Feature-Based)

### Strategy

Abandon reconstruction-based scoring entirely. PatchCore uses a **pre-trained WideResNet-50** backbone to extract patch-level features from normal training images, stores them in a **memory bank** with coreset subsampling, and detects anomalies by measuring the **nearest-neighbor distance** between test patches and the memory bank.

Key advantages over reconstruction-based methods:
- **No training required** — Uses frozen pre-trained features
- **Per-category by design** — Each category builds its own memory bank
- **Patch-level analysis** — Detects localized defects, not just global differences

### Baseline Configuration

| Parameter | Value |
|---|---|
| Backbone | WideResNet-50 (layers 2+3) |
| Resolution | 256 × 256 |
| Coreset ratio | 10% |
| Scoring | Top-1 nearest neighbor |

### Results: Mean AUROC 0.90

| Metric | Per-Category AE | PatchCore | Change |
|---|---|---|---|
| Mean AUROC | 0.7012 | **0.8975** | +0.1963 |
| Median AUROC | 0.7389 | **0.9515** | +0.2126 |
| Categories ≥ 0.95 | 1 | **11** | — |
| Categories = 1.00 | 0 | **2** | — |

**Top performers** (AUROC ≥ 0.95):

| Category | AUROC |
|---|---|
| leather | **1.0000** |
| metal_nut | **1.0000** |
| zipper | 0.9934 |
| hazelnut | 0.9917 |
| carpet | 0.9831 |
| bottle | 0.9818 |
| wood | 0.9799 |
| pipe_fryum | 0.9792 |
| chewinggum | 0.9695 |
| pcb4 | 0.9611 |
| fryum | 0.9607 |

**Weak categories** (AUROC < 0.80):

| Category | AUROC | Gap to median |
|---|---|---|
| grid | 0.5437 | −0.4078 |
| screw | 0.6390 | −0.3125 |
| capsules | 0.7858 | −0.1657 |

PatchCore achieved a **+28% relative improvement** over Per-Category AE and brought 11 of 27 categories above 0.95 AUROC.

---

## Phase 4 — Fine-Tuning: Enhanced PatchCore Sweep

### Strategy

Target the three weakest categories (grid, screw, capsules) with hyperparameter exploration. Four configurations were tested, varying resolution, feature layers, normalization, and scoring:

| Config | Resolution | Layer 1 | L2 Norm | Neighborhood | Top-K |
|---|---|---|---|---|---|
| A: 512+L1+neigh | 512 | ✓ | ✗ | 3 | 1 |
| B: 512+neigh | 512 | ✗ | ✗ | 3 | 1 |
| C: 256+L1+L2+k3 | 256 | ✓ | ✓ | 3 | 3 |
| D: 512+L1+L2+neigh | 512 | ✓ | ✓ | 3 | 1 |

### Results

| Category | Baseline | Config A | Config B | Config C | Config D | Best | Δ |
|---|---|---|---|---|---|---|---|
| **grid** | 0.5437 | 0.7540 | **0.7738** | 0.5913 | 0.5238 | **B** | **+0.2301** |
| **capsules** | 0.7858 | 0.8382 | **0.8545** | 0.6841 | 0.8106 | **B** | **+0.0687** |
| **screw** | 0.6390 | 0.6283 | 0.6361 | 0.6693 | **0.6820** | **D** | **+0.0430** |

### Key Findings

- **Config B** (512px, no Layer 1, no L2 norm) was the best overall configuration, dramatically improving grid (+0.23) and capsules (+0.07).
- **Higher resolution (512px)** is consistently beneficial for grid and capsules, where defects are spatially small.
- **L2 normalization hurts** grid detection — Config D (with L2) worsened grid from baseline, while configs without L2 (A, B) both improved it significantly.
- **Screw remains challenging** — All configs clustered around 0.63–0.68 AUROC, suggesting screw anomalies require a fundamentally different approach (e.g., rotation-invariant features or fine-grained alignment).

---

## Final Comparative Summary

### Full 27-Category Comparison

| Category | Global AE V1 | Per-Category AE | PatchCore |
|---|---|---|---|
| bottle | 0.3886 | 0.7295 | **0.9818** |
| cable | 0.6773 | 0.8448 | **0.9076** |
| candle | 0.4112 | **0.9240** | 0.8723 |
| capsule | 0.5312 | 0.8261 | **0.8582** |
| capsules | 0.5224 | 0.5850 | **0.7858** |
| carpet | 0.4662 | 0.6974 | **0.9831** |
| cashew | 0.9434 | **0.9726** | 0.8457 |
| chewinggum | 0.7515 | 0.7694 | **0.9695** |
| fryum | **0.9076** | 0.7156 | 0.9607 |
| grid | 0.4048 | **0.8810** | 0.5437 |
| hazelnut | 0.7264 | 0.9208 | **0.9917** |
| leather | 0.2714 | 0.5526 | **1.0000** |
| macaroni1 | 0.7519 | 0.7888 | **0.9199** |
| macaroni2 | **0.5540** | 0.4355 | 0.8096 |
| metal_nut | 0.2800 | 0.4745 | **1.0000** |
| pcb1 | 0.5593 | 0.7389 | **0.9515** |
| pcb2 | 0.7918 | **0.9288** | 0.8196 |
| pcb3 | 0.7324 | 0.7539 | **0.8303** |
| pcb4 | 0.7330 | 0.5474 | **0.9611** |
| pill | 0.7411 | 0.7929 | **0.8787** |
| pipe_fryum | 0.5617 | 0.8546 | **0.9792** |
| screw | **1.0000** | 0.2888 | 0.6390 |
| tile | **0.9259** | 0.5202 | 0.9562 |
| toothbrush | 0.5194 | 0.4750 | **0.8528** |
| transistor | 0.1950 | 0.2567 | **0.9600** |
| wood | 0.8697 | 0.9424 | **0.9799** |
| zipper | 0.7220 | 0.7155 | **0.9934** |
| | | | |
| **Mean** | **0.6274** | **0.7012** | **0.8975** |
| **Median** | — | **0.7389** | **0.9515** |

Bold indicates best approach per category.

### Approach Progression

| Stage | Approach | Mean AUROC | Improvement |
|---|---|---|---|
| 1. Baseline | Global AE V1 (single model, 27 categories) | 0.6274 | — |
| 2. Specialization | Per-Category AE (27 models × 30 epochs) | 0.7012 | +12% |
| 3. Paradigm shift | PatchCore (feature-based, memory bank) | 0.8975 | +28% |
| 4. Fine-tuning | Enhanced PatchCore (sweep on weak categories) | 0.9066* | +1% |

*Estimated mean after applying best sweep configs to grid (0.77), capsules (0.85), and screw (0.68).

### Key Takeaways

1. **Reconstruction-based methods are fundamentally limited** for multi-category anomaly detection. Even with per-category specialization, they rely on the model's failure to reconstruct defects — an unreliable signal for subtle anomalies.

2. **Pre-trained features outperform learned reconstruction** by a wide margin. PatchCore's WideResNet-50 backbone captures semantically meaningful features without any training.

3. **No single approach wins every category.** The Global AE V1 achieved a perfect 1.00 on screw (where PatchCore scored 0.64), and Per-Category AE led on grid (0.88 vs 0.54 for PatchCore baseline). An ensemble picking the best model per category could yield even higher performance.

4. **Resolution matters for small defects.** The sweep showed that increasing from 256px to 512px improved grid detection from 0.54 to 0.77 AUROC.

5. **Diminishing returns on hyperparameter tuning** — The sweep improved weak categories by +0.23 at best, but the remaining hard cases (screw) may require architecturally different solutions.

---

## Generated Visualizations

### Per-Model (in `outputs/<model>/evaluation/`)

| File | Description |
|---|---|
| `roc_pr_curves.png` | ROC and Precision-Recall curves (4 scorers) |
| `error_distribution.png` | Histogram of MSE for good vs anomaly |
| `auroc_per_category.png` | AUROC per category bar chart |
| `reconstructions_good.png` | Original → Reconstruction → Error Map |
| `reconstructions_anomaly.png` | Top anomaly reconstruction samples |
| `evaluation_results.json` | Full results in JSON format |

### Cross-Model Comparison (in `figures/`)

| File | Description |
|---|---|
| `comparison_global_metrics.png` | Bar chart of all global metrics |
| `comparison_auroc_heatmap.png` | Heatmap: AUROC per category × model |
| `comparison_auroc_per_category.png` | Grouped bar chart per category |
| `comparison_radar_chart.png` | Radar chart of key metrics |
| `comparison_confusion_matrices.png` | Side-by-side confusion matrices |
| `comparison_summary_table.png` | Summary table with rankings |

---

## Known Limitations & Future Directions

1. **Screw remains unsolved** — AUROC 0.64–0.68 across all configurations. Thread-like structure may require rotation-invariant features or geometric alignment preprocessing.
2. **Grid baseline is low** — Improved to 0.77 with higher resolution but still below the dataset median. Periodic textures need specialized frequency-domain analysis.
3. **No pixel-level localization** — Current evaluation is image-level only. PatchCore natively supports anomaly heatmaps; adding localization metrics (pixel-AUROC, PRO) would strengthen the analysis.
4. **Ensemble potential untapped** — Combining the best model per category (AE for screw, PatchCore for most others) could push mean AUROC above 0.93.
5. **Resolution constraint** — All reconstruction models trained at 256×256. High-resolution images (up to 1562×960 in VisA) lose detail when downscaled.

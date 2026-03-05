# Results & Analysis

Experimental results from training and evaluating all models on the combined MVTec AD + VisA dataset (27 categories, 15,218 images).

## Training Configuration

| Detail | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, Ada Lovelace) |
| **PyTorch** | 2.6.0+cu124 |
| **Dataset** | 12,050 training images, 3,168 test images (1,667 good + 1,501 anomaly) |
| **Categories** | 27 (15 MVTec AD + 12 VisA) |
| **Image size** | 256 × 256 RGB |

## Global Performance Summary

| Model | AUROC (MSE) | AUROC (SSIM) | AUROC (Perc.) | AUROC (Comb.) | AP (MSE) | AP (Comb.) | Rank |
|---|---|---|---|---|---|---|---|
| **Autoencoder V1** | **0.5397** | **0.5485** | **0.5286** | **0.5339** | **0.5534** | **0.5290** | **#1 (55.1 pts)** |
| Autoencoder V2 | 0.5245 | 0.5286 | 0.5253 | 0.5245 | 0.5150 | 0.5243 | #2 (40.1 pts) |
| GAN | 0.5351 | 0.5165 | 0.5088 | 0.5148 | 0.5064 | 0.4874 | #3 (31.1 pts) |
| Diffusion (DDPM) | 0.4960 | 0.4916 | 0.4832 | 0.4835 | 0.4914 | 0.4637 | #4 (15.7 pts) |

## Training Loss Progression

| Model | Epochs | Initial Loss | Final Loss | Data Used |
|---|---|---|---|---|
| Autoencoder V1 | 50 | 0.0096 | 0.00036 | 12,050 images |
| Autoencoder V2 | 50 | 0.1853 | 0.0510 | 12,050 images (augmented) |
| GAN | 50 | G=4.86, D=0.69 | G=1.56, D=0.41 | 12,050 images |
| Diffusion | 40 | 0.1975 | 0.0162 | 12,050 images |

## Reconstruction Error Separation

| Model | MSE Good (mean) | MSE Anomaly (mean) | MSE Gap | SSIM Good | SSIM Anomaly |
|---|---|---|---|---|---|
| AE V1 | 0.000330 | 0.000367 | +0.000037 | 0.9574 | 0.9541 |
| AE V2 | 0.001868 | 0.001979 | +0.000111 | 0.8540 | 0.8466 |
| GAN | 0.002447 | 0.002585 | +0.000138 | 0.7988 | 0.7967 |
| Diffusion | 0.057427 | 0.057255 | −0.000172 | 0.1361 | 0.1359 |

## Top-Performing Categories (AUROC Combined, best model)

| Category | Best Model | AUROC (Comb.) |
|---|---|---|
| screw | AE V1 | **1.0000** |
| cashew | AE V1 | 0.9434 |
| wood | AE V2 | 0.9248 |
| tile | AE V1 | 0.9259 |
| screw | AE V2 | 0.9112 |
| fryum | AE V1 | 0.9076 |

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

## Analysis & Discussion

### Why AUROC values are close to 0.5

All four reconstruction-based models achieve global AUROC values in the 0.48–0.55 range, which is only marginally above random chance. This is a known challenge with reconstruction-based anomaly detection on diverse multi-category datasets:

1. **Single model across 27 categories**: Training one model on products as diverse as screws, PCBs, and candles forces the model to learn a very general representation. Anomalies in one category (e.g., scratches on metal) may look like normal patterns in another (e.g., textures on wood).

2. **Small MSE separation**: The gap between good and anomaly MSE means is tiny (e.g., 0.000037 for AE V1). Both good and anomaly images are reconstructed at similar quality because the model only learned population-level statistics, not fine-grained defect patterns.

3. **Category heterogeneity cancels out**: Some categories achieve excellent AUROC (screw: 1.00, cashew: 0.94) while others are near random or inverted (transistor: 0.19, leather: 0.27). When aggregated globally, these cancel out toward 0.5.

4. **Diffusion's negative separation**: The diffusion model actually reconstructs anomalies slightly *better* than good images (MSE gap = −0.000172), leading to below-random global AUROC. This happens because the single-step denoising at t=250 is too coarse to preserve fine details that distinguish anomalies.

### Model-Specific Observations

- **Autoencoder V1** wins overall because its wider bottleneck (65K dims) preserves enough information for good categories while still creating separation in structured objects (screw, tile, cashew).

- **Autoencoder V2** has better MSE separation (0.000111 vs 0.000037) thanks to the tighter bottleneck (8K dims, 24:1 compression), but the aggressive compression also degrades reconstruction quality across the board (SSIM 0.854 vs 0.957), hurting some categories.

- **GAN** shows the best absolute MSE separation (0.000138) but adversarial training instability (D_loss oscillating 0.40–0.47) limits reconstruction consistency.

- **Diffusion** is the worst performer for anomaly detection because the denoising process doesn't create meaningful reconstruction error differences. More timesteps or multi-scale denoising would be needed.

### Path to Better Performance

The primary bottleneck is the **single-model-for-all-categories** approach. State-of-the-art methods achieve AUROC > 0.95 on MVTec AD by:

1. **Per-category models** — Train 27 separate models, each specialized for one product type
2. **Feature-based methods** — PatchCore, STPM, EfficientAD extract pre-trained features rather than training from scratch
3. **Memory banks** — Store prototypical normal features and detect anomalies by nearest-neighbor distance
4. **Larger models** — WideResNet or EfficientNet backbones with 10-100× more parameters

---

## Known Limitations & Future Work

### Current Limitations

1. **Single generic model across 27 categories** — Training one model on all categories dilutes specialization. Per-category models would likely achieve much higher AUROC.
2. **Low global discriminability** — AUROC ~0.53 (best) indicates models struggle to separate good from anomaly at the population level.
3. **Variable-size input dataset** — Images range from 224×224 to 1562×960, all resized to 256×256, potentially losing critical defect details in high-resolution images.
4. **Diffusion denoising strategy** — Single-step denoising at t=250 is too coarse for fine anomaly detection.

### Potential Improvements

- **Multi-scale evaluation** — Analyze reconstruction error at multiple resolutions to catch both coarse and fine anomalies.
- **Ensemble scoring** — Combine predictions from multiple models for more robust anomaly detection.
- **Higher resolution** — Train at 512×512 or native resolution to preserve defect details.
- **Latent space methods** — VAE with KL divergence anomaly scoring in latent space.
- **Better diffusion strategy** — Multi-step denoising, multiple noise levels, or Noise2Score approach.

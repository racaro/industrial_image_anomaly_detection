# Usage Guide

Detailed instructions for training, evaluating, and comparing all models in the anomaly detection pipeline.

## Pipeline Overview

### Training

All training scripts share common utilities from `src/` and automatically detect GPU (CUDA) when available.

#### Autoencoder V1 (`python -m src.models.autoencoder.train`)

1. **Dataset Exploration** — Scan all 27 categories, count images per split.
2. **Balancing** — If a category has 0 test/good images, move images from train/good → test/good.
3. **Image Validation** — Verify every image can be opened; report format/size distributions.
4. **Training** — 50 epochs with `MSELoss` + `Adam` (lr=1e-3, batch_size=32) on all 12,050 images.
5. **Export** — Save model weights to `outputs/autoencoder/model.pth`.

#### Autoencoder V2 (`python -m src.models.autoencoder.train_v2`)

1. Same dataset pipeline as V1 (exploration, validation).
2. **Data Augmentation** — Random horizontal/vertical flips, rotation (±15°), color jitter, random affine.
3. **Combined Loss** — `MSE + 0.3 × (1 − SSIM)`.
4. **Training** — 50 epochs with `AdamW` (lr=2e-4, wd=1e-4), cosine annealing LR, gradient clipping.
5. **Export** — Save model to `outputs/autoencoder_v2/model.pth`.

#### GAN (`python -m src.models.gan.train`)

1. Same dataset pipeline as V1.
2. **Adversarial Training** — Alternating D/G optimization:
   - **D step**: minimize `BCE(D(real), 1) + BCE(D(G(real)), 0)`
   - **G step**: minimize `λ_adv × BCE(D(G(real)), 1) + λ_rec × MSE(real, G(real))`
3. **Training** — 50 epochs with separate Adam optimizers (lr_G=lr_D=1e-4).
4. **Export** — Save Generator + Discriminator weights.

#### Diffusion DDPM (`python -m src.models.diffusion.train`)

1. Same dataset pipeline.
2. **Forward Diffusion** — Add Gaussian noise over 1000 timesteps (cosine β schedule).
3. **Training** — UNet learns to predict noise ε at random timestep t. 40 epochs with `AdamW` (lr=2e-4), cosine annealing LR.
4. **Anomaly Scoring** — At evaluation, apply noise at timestep 250 and denoise; reconstruction error indicates anomaly.
5. **Export** — Save model to `outputs/diffusion/model.pth`.

#### Per-Category Autoencoders (`python -m src.models.autoencoder.train_per_category`)

1. Trains one independent AE V1 per category (27 models).
2. Each model specializes on a single product type's normal appearance.
3. **Export** — Save models to `outputs/autoencoder_per_category/<category>/model.pth`.

#### PatchCore (`python -m src.models.patchcore.build_memory_bank`)

1. No training — uses frozen WideResNet-50 features.
2. **Feature Extraction** — Extract multi-scale patch features (layers 2 & 3) from train/good images.
3. **Coreset Subsampling** — Greedy farthest-point sampling reduces memory bank to 10% of patches.
4. **Export** — Save memory banks to `outputs/patchcore/<category>/memory_bank.pt`.

### Evaluation (`src/evaluate.py`)

A single unified evaluation script supports all 4 reconstruction-based models via `--model` flag:

1. **Load Model** — Load the chosen model's weights.
2. **Collect Test Images** — Gather test/good (label=0) and test/anomaly (label=1) images from 27 categories.
3. **Compute Pixel Metrics** — MSE, MAE, and SSIM between original and reconstruction.
4. **Compute Perceptual Score** — VGG-16 feature distance between original and reconstruction.
5. **Compute Combined Score** — `0.3 × MSE_norm + 0.3 × SSIM_norm + 0.4 × Perceptual_norm` (min-max normalized).
6. **Global Metrics** — AUROC and Average Precision using MSE, SSIM, Perceptual, and Combined scores.
7. **Per-Category Metrics** — AUROC and AP per category using Combined score.
8. **Optimal Threshold** — Youden's J statistic on the Combined ROC curve.
9. **Visualizations** — ROC/PR curves, error distributions, AUROC bar chart, reconstruction samples.
10. **Export** — CSV, JSON, and PNG files to `outputs/<model>/evaluation/`.

### Model Comparison (`src/compare_models.py`)

Generates 9 comparison visualizations across all models:
- Global metrics bar chart, AUROC heatmap, per-category AUROC chart
- Radar chart, confusion matrices, SSIM & error distributions
- Summary table with weighted ranking system

---

## Commands

### Train Models

```bash
# Autoencoder V1
python -m src.models.autoencoder.train

# Autoencoder V2 (with augmentation + combined loss)
python -m src.models.autoencoder.train_v2

# Per-Category Autoencoders (27 independent models)
python -m src.models.autoencoder.train_per_category

# GAN
python -m src.models.gan.train

# Diffusion (DDPM)
python -m src.models.diffusion.train

# PatchCore (build memory banks — no training needed)
python -m src.models.patchcore.build_memory_bank
```

### Evaluate

```bash
python -m src.evaluate --model autoencoder
python -m src.evaluate --model autoencoder_v2
python -m src.evaluate --model gan
python -m src.evaluate --model diffusion

# Per-category autoencoders
python -m src.evaluate_per_category --compare-with autoencoder

# PatchCore
python -m src.evaluate_patchcore --compare-with autoencoder
```

Results (plots, CSV, JSON) are saved to `outputs/<model>/evaluation/`.

### Compare

```bash
# Compare reconstruction-based models
python -m src.compare_models

# Compare all approaches (global, per-category, PatchCore)
python -m src.compare_all_approaches
```

Generates comparison charts in `figures/`.

### Anomaly Localization

```bash
# Generate heatmaps for specific categories
python -m src.localization --categories bottle leather metal_nut

# All categories
python -m src.localization --max-per-category 10
```

Heatmaps are saved to `outputs/patchcore/localization/`.

### Enhanced PatchCore (Weak Categories)

```bash
# Run enhanced pipeline
python -m src.models.patchcore.enhanced_features --categories grid screw capsules

# Configuration sweep
python -m src.models.patchcore.enhanced_features --sweep --categories grid screw capsules

# Apply best configs
python -m src.models.patchcore.apply_best_configs
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

Shared hyperparameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_HEIGHT` / `IMG_WIDTH` | 256 | Input image resolution |
| `BATCH_SIZE` | 32 | Training and evaluation batch size |
| `NUM_EPOCHS` | 50 | Number of training epochs (AE V1, GAN) |
| `LEARNING_RATE` | 1e-3 | Autoencoder V1 Adam learning rate |
| `DEVICE` | auto | `cuda` if available, else `cpu` |

Model-specific hyperparameters are in each model's `train.py`:

| Model | Key Hyperparameters |
|---|---|
| AE V2 | lr=2e-4, wd=1e-4, SSIM weight=0.3, cosine LR |
| GAN | λ_adv=1.0, λ_rec=50.0, lr_G=lr_D=1e-4 |
| Diffusion | lr=2e-4, 1000 timesteps, cosine β schedule, 40 epochs |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error between original and reconstructed image (pixel-level) |
| **MAE** | Mean Absolute Error between original and reconstructed image |
| **SSIM** | Structural Similarity Index — measures perceptual similarity |
| **Perceptual** | VGG-16 multi-layer feature distance (L2 in feature space) |
| **Combined** | Weighted fusion: `0.3×MSE + 0.3×SSIM + 0.4×Perceptual` (min-max normalized) |
| **AUROC** | Area Under the ROC Curve — measures binary classification quality |
| **Average Precision** | Area under the Precision-Recall curve — robust to class imbalance |
| **Youden's J** | Optimal threshold for classification (`argmax(TPR − FPR)`) |

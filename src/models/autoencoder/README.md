# Convolutional Autoencoder

Symmetric encoder-decoder for unsupervised anomaly detection. Trained only on defect-free images; anomalies are detected via high reconstruction error at inference time. Three variants are implemented: **V1** (baseline), **V2** (improved), and **Per-Category** (specialized).

## Architecture

### Autoencoder V1 (~4.4M parameters)

Standard 4-layer convolutional autoencoder. Latent space: 256×16×16 = 65,536 dims (compression ratio ~3:1).

```
Input (3 × 256 × 256)
        │
        ▼
┌─── ENCODER ───────────────────────────────────┐
│  Conv2d(3→64,   k=4, s=2, p=1) + BN + ReLU   │  → 64 × 128 × 128
│  Conv2d(64→128,  k=4, s=2, p=1) + BN + ReLU   │  → 128 × 64 × 64
│  Conv2d(128→256, k=4, s=2, p=1) + BN + ReLU   │  → 256 × 32 × 32
│  Conv2d(256→256, k=4, s=2, p=1) + BN + ReLU   │  → 256 × 16 × 16
└────────────────────────────────────────────────┘
        │  Latent: 256 × 16 × 16 = 65,536 dims
        ▼
┌─── DECODER ───────────────────────────────────┐
│  ConvT2d(256→256, k=4, s=2, p=1) + BN + ReLU  │  → 256 × 32 × 32
│  ConvT2d(256→128, k=4, s=2, p=1) + BN + ReLU  │  → 128 × 64 × 64
│  ConvT2d(128→64,  k=4, s=2, p=1) + BN + ReLU  │  → 64 × 128 × 128
│  ConvT2d(64→3,    k=4, s=2, p=1) + Sigmoid     │  → 3 × 256 × 256
└────────────────────────────────────────────────┘
        │
        ▼
Output (3 × 256 × 256)
```

### Autoencoder V2 (~2.4M parameters)

Improved 5-layer architecture. Latent space: 128×8×8 = 8,192 dims (compression ratio ~24:1).

```
Input (3 × 256 × 256)
        │
        ▼
┌─── ENCODER ───────────────────────────────────────────┐
│  Conv2d(3→32,    k=4, s=2, p=1) + BN + LeakyReLU     │  → 32 × 128 × 128
│  Conv2d(32→64,   k=4, s=2, p=1) + BN + LeakyReLU     │  → 64 × 64 × 64
│  Conv2d(64→128,  k=4, s=2, p=1) + BN + LeakyReLU     │  → 128 × 32 × 32
│  Conv2d(128→256, k=4, s=2, p=1) + BN + LeakyReLU     │  → 256 × 16 × 16
│  Conv2d(256→128, k=4, s=2, p=1) + BN + LeakyReLU     │  → 128 × 8 × 8
│  Dropout2d(p=0.1)                                      │
└────────────────────────────────────────────────────────┘
        │  Latent: 128 × 8 × 8 = 8,192 dims
        ▼
┌─── DECODER (mirrors encoder) ─────────────────────────┐
│  ConvT2d(128→256) + BN + LeakyReLU                    │  → 256 × 16 × 16
│  ConvT2d(256→128) + BN + LeakyReLU                    │  → 128 × 32 × 32
│  ConvT2d(128→64)  + BN + LeakyReLU                    │  → 64 × 64 × 64
│  ConvT2d(64→32)   + BN + LeakyReLU                    │  → 32 × 128 × 128
│  ConvT2d(32→3)    + Sigmoid                            │  → 3 × 256 × 256
└────────────────────────────────────────────────────────┘
```

### V1 vs V2 Comparison

| Feature | V1 | V2 |
|---|---|---|
| Layers | 4 encoder + 4 decoder | 5 encoder + 5 decoder |
| Latent dim | 65,536 (3:1) | 8,192 (24:1) |
| Activation | ReLU | LeakyReLU |
| Regularization | None | Dropout2d at bottleneck |
| Initialization | Default | Kaiming |
| Loss | MSE | MSE + 0.3×(1−SSIM) |
| Optimizer | Adam (lr=1e-3) | AdamW (lr=2e-4, wd=1e-4) |
| LR schedule | None | CosineAnnealingLR |
| Augmentation | None | Flips, rotation, color jitter, affine |

## Training

### V1 — Global model

```bash
python -m src.models.autoencoder.train
```

| Parameter | Value |
|---|---|
| Loss | MSELoss |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 50 |
| Batch size | 32 |
| Data | 12,050 train/good images (all categories) |

Weights → `outputs/autoencoder/model.pth`

### V2 — Global model with improvements

```bash
python -m src.models.autoencoder.train_v2
```

| Parameter | Value |
|---|---|
| Loss | MSE + 0.3×(1−SSIM) |
| Optimizer | AdamW (lr=2e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Augmentation | HFlip, VFlip, Rotation(±15°), ColorJitter, RandomAffine |
| Gradient clipping | max_norm=1.0 |
| Epochs | 50 |

Weights → `outputs/autoencoder_v2/model.pth`

### Per-Category — One V1 model per category

```bash
# Train all 27 categories (default 30 epochs)
python -m src.models.autoencoder.train_per_category

# Custom epochs
python -m src.models.autoencoder.train_per_category --epochs 50

# Specific categories only
python -m src.models.autoencoder.train_per_category --categories bottle cable screw
```

| Parameter | Value |
|---|---|
| Architecture | Same as V1 |
| Loss | MSELoss |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 30 (default) |
| Models | 27 independent models |

Weights → `outputs/autoencoder_per_category/<category>/model.pth`

## Anomaly Scoring

At test time, the anomaly score for an image $x$ is:

$$\text{score}(x) = \text{MSE}(x,\; f(x))$$

where $f$ is the trained autoencoder. Higher score → more likely anomalous.

The unified evaluation pipeline also computes SSIM, VGG-16 perceptual distance, and a combined score:

$$\text{Combined} = 0.3 \times \text{MSE}_\text{norm} + 0.3 \times \text{SSIM}_\text{norm} + 0.4 \times \text{Perceptual}_\text{norm}$$

## Evaluation

```bash
# Global V1
python -m src.evaluate --model autoencoder

# Global V2
python -m src.evaluate --model autoencoder_v2

# Per-category (compares with global model)
python -m src.evaluate_per_category --compare-with autoencoder
```

Results are saved to `outputs/<model>/evaluation/`.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `padding=1` on all layers | Prevents border artifacts that occur with `padding=0` |
| Strided convolutions (`stride=2`) | Learnable downsampling (replaces pooling) |
| BatchNorm after every conv | Stabilizes and accelerates training |
| Sigmoid output | Keeps reconstructions in `[0, 1]` matching the input range |
| Tighter bottleneck in V2 | Forces model to learn compact representations, better anomaly separation |
| Per-category training | Specializes each model for one product type, avoids cross-category interference |

## Module Structure

| File | Description |
|---|---|
| `model.py` | Autoencoder V1 architecture (4-layer, 65K latent) |
| `model_v2.py` | Autoencoder V2 architecture (5-layer, 8K latent, dropout) |
| `train.py` | V1 training pipeline |
| `train_v2.py` | V2 training pipeline (augmentation + combined loss) |
| `train_per_category.py` | Per-category V1 training (27 independent models) |

## References

- Bergmann, P. et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
- An, J. & Cho, S. *Variational Autoencoder based Anomaly Detection using Reconstruction Probability.* 2015.
- Wang, Z. et al. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004 (SSIM loss in V2).

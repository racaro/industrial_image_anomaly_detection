# Anomaly Detection on Industrial Images

Unsupervised anomaly detection pipeline comparing **reconstruction-based** models (Autoencoder, GAN, Diffusion) with a **feature-based** approach (PatchCore), trained and evaluated on 27 product categories from the **MVTec AD** and **VisA** benchmarks.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Results Preview](#results-preview)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Documentation](#documentation)
9. [License](#license)

---

## Problem Statement

Manufacturing quality control relies on visual inspection to detect defective products. Manual inspection is slow, error-prone, and does not scale. This project implements **unsupervised anomaly detection**: models learn what "normal" looks like from defect-free images only, then flag anything that deviates from that learned representation.

Two paradigms are compared:

- **Reconstruction-based** — Autoencoder, GAN, and Diffusion models learn to reconstruct normal images. Anomalies produce higher reconstruction error.
- **Feature-based** — PatchCore uses pre-trained ImageNet features and nearest-neighbor distance. No model training required.

---

## Dataset

The `combined_dataset/` directory is built by manually merging two public benchmarks into a unified folder structure. The images are **not included in the repository** — you must download each dataset from its original source and organize them following the layout below.

### Sources

| Benchmark | Categories | Source |
|---|---|---|
| **MVTec AD** | 15 (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper) | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | 12 (candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum) | [kaggle.com/ess1004/visa-anomaly-detection](https://www.kaggle.com/datasets/ess1004/visa-anomaly-detection) |

### Combined Statistics

| Split | Count |
|---|---|
| Train / Good | 12,050 |
| Test / Good | 1,667 |
| Test / Anomaly | 1,501 |
| **Total** | **15,218** |

### Directory Layout

After downloading both datasets, organize them into this structure at the project root:

```
combined_dataset/
├── bottle/                    # MVTec AD category
│   ├── train/
│   │   └── good/              # defect-free images only
│   └── test/
│       ├── good/              # defect-free test images
│       └── anomaly/           # defective test images
├── candle/                    # VisA category
│   ├── train/good/
│   └── test/{good,anomaly}/
├── ...                        # 25 more categories
└── zipper/
    ├── train/good/
    └── test/{good,anomaly}/
```

All images are RGB. During training, every image is resized to **256 × 256** and normalized to `[0, 1]`.

---

## Architecture

Five approaches across two paradigms:

| Model | Params | Approach | Training Signal | Details |
|---|---|---|---|---|
| **Autoencoder V1** | ~4.4M | Reconstruction (3:1 compression) | MSE | [README](src/models/autoencoder/README.md) |
| **Autoencoder V2** | ~2.4M | Reconstruction (24:1 compression) | MSE + SSIM | [README](src/models/autoencoder/README.md) |
| **GAN** | ~7.2M (G+D) | Adversarial reconstruction | BCE + MSE | [README](src/models/gan/README.md) |
| **Diffusion (DDPM)** | ~2.7M | Denoising (ε-prediction) | MSE | [README](src/models/diffusion/README.md) |
| **PatchCore (baseline)** | 68.9M (frozen) | Feature-based k-NN | Pre-trained ImageNet | [README](src/models/patchcore/README.md) |
| **PatchCore (enhanced)** | 68.9M (frozen) | Feature-based k-NN + 4 improvements | Pre-trained ImageNet | [Enhanced features](src/models/patchcore/enhanced_features.py) |

### PatchCore Variants

**Baseline PatchCore** (`src/models/patchcore/build_memory_bank.py`):
- Multi-scale patch features from WideResNet-50 layers 2 & 3
- Concatenated into 1536-dim descriptors per spatial position
- Per-category memory banks (10% coreset subsampling)

**Enhanced PatchCore** (`src/models/patchcore/enhanced_features.py`):
- Adds Layer 1 features (256 channels, stride 4) → **+256 dims = 1792 total**
- Local neighborhood aggregation (3×3 avg pooling) → robust to small spatial shifts
- L2 feature normalization → balanced distance metrics
- Higher resolution support (512×512 input) → 64×64 spatial grid (4× more patches)

Enhanced variant specifically targets weak categories (grid, screw, capsules) with configurable `EnhancedConfig`.

Each model README includes architecture diagrams, hyperparameters, design decisions, and references.

---

## Results Preview

### Global Performance Comparison

| Model | Approach | AUROC (Combined) | AUROC (MSE) | Mean AUROC/Category | Categories ≥ 0.9 |
|---|---|---|---|---|---|
| **Autoencoder V1** | Reconstruction (MSE) | 0.5339 | 0.5397 | 0.627 | 4 |
| **Autoencoder V2** | Reconstruction (MSE + SSIM) | 0.5245 | 0.5245 | 0.618 | 2 |
| **GAN** | Adversarial Reconstruction | 0.5148 | 0.5351 | 0.618 | 2 |
| **Diffusion (DDPM)** | Denoising-based | 0.4835 | 0.4960 | 0.540 | 2 |
| **Per-Category Autoencoder** | Per-category Reconstruction | — | — | **0.701** | 10 |
| **PatchCore (baseline)** | Feature-based (k-NN) | — | — | **0.897** | **22** |
| **PatchCore (enhanced)** | Feature-based + Layer1 + L2 norm | — | — | **0.909** | **23** |

**Key Finding**: PatchCore achieves **0.897 mean AUROC**. Enhanced variant improves weak categories (+1.3%), reaching **0.909** mean when applied selectively (grid: +0.23, screw: +0.04, capsules: +0.02).

### Per-Category Performance: Baseline vs Enhanced

Enhanced PatchCore targets 3 weak categories (grid, screw, capsules):

| Category | Baseline AUROC | Enhanced AUROC | Δ | Status |
|---|---|---|---|---|
| **grid** | 0.5437 | 0.7700 | +0.2263 | ✓ Major improvement |
| **screw** | 0.6390 | 0.6820 | +0.0430 | ✓ Modest gain |
| **capsules** | 0.7858 | 0.8106 | +0.0248 | ✓ Modest gain |

Enhanced features especially help grid (4× more patches capture fine textures) and screw (layer 1 captures thread patterns).

### Per-Category AUROC Breakdown (Baseline PatchCore)

Baseline performance across all 27 categories (sorted):

| Excellent (≥ 0.95) | Strong (0.85–0.95) | Moderate (0.70–0.85) | Weak (< 0.70) |
|---|---|---|---|
| leather (1.00), metal_nut (1.00), hazelnut (0.99), zipper (0.99), bottle (0.98), carpet (0.98), pipe_fryum (0.98), chewinggum (0.97), transistor (0.96), pcb1 (0.95) | fryum (0.96), pcb4 (0.96), tile (0.96), cable (0.91), macaroni1 (0.92), wood (0.98), candle (0.87), capsule (0.86), pill (0.88), toothbrush (0.85) | pcb2 (0.82), pcb3 (0.83), macaroni2 (0.81), capsules (0.79) | grid (0.54), screw (0.64) |

### Why Reconstruction Models Failed; Why PatchCore Succeeded

**Reconstruction paradigm limitations**:
1. Single global model ≈ random detection (AUROC 0.5): Training on 27 heterogeneous categories forces generic representations
2. Per-category specialization helps (+12%): Using 27 independent models improves to AUROC 0.70, but still below feature-based approach
3. Fundamental issue: Defects don't always produce *different* reconstruction error than normal images

**Feature-based advantages**:
1. Pre-trained ImageNet backbone captures diverse visual concepts without any module training
2. k-NN in feature space is robust to category-specific appearance variations
3. No need to learn what anomalies look like — inherently captured by outlier detection in feature space

### Training Details

- **GPU**: NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, Ada Lovelace)
- **Framework**: PyTorch 2.6.0+cu124, Python 3.10+
- **Dataset**: MVTec AD (15 categories) + VisA (12 categories) = 27 total, 12,050 training images, 3,168 test images
- **Image size**: 256 × 256 RGB (baseline), 512 × 512 (enhanced), normalized to [0, 1]

For detailed analysis, experimental progression, per-category breakdowns, and ablation studies, see [docs/RESULTS.md](docs/RESULTS.md).

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA 12.x (recommended — GPU training is 5-10× faster)

### Setup

```bash
git clone <repo-url>
cd anomaly_detection_industrial_images

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://www.kaggle.com/datasets/ess1004/visa-anomaly-detection) datasets, then organize them into the `combined_dataset/` structure described [above](#directory-layout). The pipeline auto-detects all categories present in this folder.

---

## Quick Start

```bash
# Train models
python -m src.models.autoencoder.train          # AE V1
python -m src.models.autoencoder.train_v2        # AE V2
python -m src.models.gan.train                   # GAN
python -m src.models.diffusion.train             # Diffusion
python -m src.models.patchcore.build_memory_bank # PatchCore

# Evaluate
python -m src.evaluate --model autoencoder
python -m src.evaluate_patchcore

# Compare all approaches
python -m src.compare_all_approaches

# Run tests
python -m pytest tests/ -v
```

See [docs/USAGE.md](docs/USAGE.md) for the full command reference including per-category training, anomaly localization, enhanced PatchCore, and configuration details.

---

## Project Structure

```
anomaly_detection_industrial_images/
├── src/
│   ├── config.py                     # Paths, hyperparameters, device config
│   ├── dataset.py                    # Dataset loading, validation, PyTorch datasets
│   ├── metrics.py                    # SSIM computation
│   ├── feature_extractor.py          # VGG-16 perceptual scoring
│   ├── evaluate.py                   # Unified model evaluation
│   ├── evaluate_per_category.py      # Per-category AE evaluation
│   ├── evaluate_patchcore.py         # PatchCore evaluation
│   ├── compare_models.py             # Cross-model comparison
│   ├── compare_all_approaches.py     # All approaches comparison
│   ├── localization.py               # Anomaly heatmap localization
│   └── models/
│       ├── autoencoder/              # AE V1, V2 & per-category
│       ├── gan/                      # Generator + PatchGAN Discriminator
│       ├── diffusion/                # DDPM U-Net
│       └── patchcore/                # Memory bank + enhanced features
├── tests/                            # Unit tests (53 tests)
├── docs/                             # Extended documentation
│   ├── USAGE.md                      # Full command reference (git-ignored)
│   └── RESULTS.md                    # Experimental results & analysis
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── combined_dataset/                 # Images (git-ignored, see Dataset section)
├── outputs/                          # Trained weights & evaluations (git-ignored)
└── figures/                          # Comparison plots (git-ignored)
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Full pipeline details, all commands, configuration tables, evaluation metrics |
| [docs/RESULTS.md](docs/RESULTS.md) | Experimental results, performance tables, analysis & discussion, limitations |
| [docs/PATCHCORE_ARCHITECTURE.md](docs/PATCHCORE_ARCHITECTURE.md) | PatchCore variants, inheritance design, enhanced features for weak categories |
| [src/models/autoencoder/README.md](src/models/autoencoder/README.md) | Autoencoder V1, V2 & per-category architecture and training |
| [src/models/gan/README.md](src/models/gan/README.md) | GAN architecture, adversarial training strategy |
| [src/models/diffusion/README.md](src/models/diffusion/README.md) | DDPM U-Net, noise schedules, inference strategy |
| [src/models/patchcore/README.md](src/models/patchcore/README.md) | PatchCore pipeline, memory banks, enhanced features |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## License

This project is for educational and research purposes. The MVTec AD and VisA datasets have their own respective licenses — please refer to the original sources for terms of use.

# Anomaly Detection on Industrial Images

Unsupervised anomaly detection pipeline using a **Convolutional Autoencoder** and a **Reconstruction-based GAN** (PyTorch), trained on the **MVTec AD** and **VisA** benchmark datasets. Both models are evaluated with identical metrics for a fair comparison.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Datasets](#datasets)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [Pipeline Overview](#pipeline-overview)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results](#results)
10. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Problem Statement

Manufacturing and industrial quality control heavily rely on visual inspection to detect defective products. Manual inspection is slow, error-prone, and does not scale. **Anomaly detection** aims to automatically identify defective items by learning what "normal" looks like and flagging anything that deviates from that learned representation.

This project implements an **unsupervised** approach: a convolutional autoencoder is trained **only on defect-free ("good") images**. At inference time, defective images produce higher reconstruction error because the model has never learned to reconstruct anomalous patterns. By thresholding the reconstruction error (MSE), we can classify images as *good* or *anomaly*.

---

## Datasets

We combine two widely-used industrial anomaly detection benchmarks into a single unified dataset (`combined_dataset/`):

### MVTec Anomaly Detection (MVTec AD)

| Property | Value |
|---|---|
| **Source** | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **Categories** | 15 (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper) |
| **Image size** | 224 × 224 px |
| **Format** | PNG |
| **Domain** | Textures and objects |

### Visual Anomaly (VisA)

| Property | Value |
|---|---|
| **Source** | [VisA Dataset](https://github.com/amazon-science/spot-diff) |
| **Categories** | 12 (candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum) |
| **Image size** | Variable (1274–1562 px wide) |
| **Format** | JPEG |
| **Domain** | Complex objects and PCB boards |

### Combined Statistics

| Split | Count |
|---|---|
| Train / Good | 12,050 |
| Test / Good | 1,667 |
| Test / Anomaly | 1,501 |
| **Total** | **15,218** |

All images are RGB. During training, every image is resized to **256 × 256** and normalized to `[0, 1]`.

### Dataset Directory Layout

```
combined_dataset/
├── bottle/
│   ├── train/
│   │   └── good/          # defect-free images
│   └── test/
│       ├── good/          # defect-free test images
│       └── anomaly/       # defective test images
├── cable/
│   ├── train/good/
│   └── test/{good,anomaly}/
├── ...
└── zipper/
    ├── train/good/
    └── test/{good,anomaly}/
```

---

## Project Structure

```
anomaly_detection_industrial_images/
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Paths, hyperparameters, device
│   ├── dataset.py                    # Dataset exploration, validation, PyTorch datasets
│   ├── metrics.py                    # SSIM computation
│   ├── evaluate.py                   # Unified evaluation (--model autoencoder|gan)
│   └── models/                       # Model definitions, training & documentation
│       ├── __init__.py               # Re-exports Autoencoder, Generator, Discriminator
│       ├── autoencoder/
│       │   ├── __init__.py
│       │   ├── model.py              # Convolutional Autoencoder
│       │   ├── train.py              # AE training pipeline
│       │   └── README.md             # AE architecture & training details
│       └── gan/
│           ├── __init__.py
│           ├── model.py              # Generator + PatchGAN Discriminator
│           ├── train.py              # GAN training pipeline
│           └── README.md             # GAN architecture & training details
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
├── combined_dataset.zip              # Dataset archive (tracked in git)
├── combined_dataset/                 # Extracted images (git-ignored)
├── outputs/                          # Trained weights & evaluations (git-ignored)
│   ├── autoencoder/
│   │   ├── model.pth
│   │   └── evaluation/
│   └── gan/
│       ├── generator.pth
│       ├── discriminator.pth
│       └── evaluation/
└── figures/                          # Generated plots (git-ignored)
```

---

## Architecture

Two reconstruction-based models are implemented. Both share the same encoder-decoder backbone so that performance differences come from the training strategy, not the model capacity.

| Model | Training Signal | Anomaly Score | Details |
|---|---|---|---|
| **Autoencoder** | MSE reconstruction loss | `MSE(x, f(x))` | [src/models/autoencoder/](src/models/autoencoder/README.md) |
| **GAN** | Adversarial + MSE reconstruction loss | `MSE(x, G(x))` | [src/models/gan/](src/models/gan/README.md) |

### Shared Design Decisions

- **`padding=1`** on all convolutional layers prevents border artifacts.
- **Strided convolutions** (`stride=2`) replace pooling for learnable downsampling.
- **Batch normalization** after each convolution stabilizes training.
- **Sigmoid** output keeps reconstructions in `[0, 1]`.
- **Identical encoder-decoder** in both AE and Generator (4 encoder + 4 decoder layers, latent 256×16×16).

See each model's dedicated README for full architecture diagrams, loss formulations, and hyperparameters.

---

## Pipeline Overview

### Training

Both training scripts share common utilities from `src/`:

#### Autoencoder (`src/models/autoencoder/train.py`)

1. **Dataset Exploration** — Scan all categories, count images per split.
2. **Balancing** — If a category has 0 test/good images, move 100 images from train/good → test/good.
3. **Image Validation** — Verify every image can be opened and report distributions.
4. **Training** — 30 epochs with `MSELoss` + `Adam` (lr=1e-3, batch_size=32).
5. **Export** — Save model weights to `outputs/autoencoder/model.pth`.

#### GAN (`src/models/gan/train.py`)

1. Same dataset pipeline as the Autoencoder (exploration, balancing, validation).
2. **Adversarial Training** — Alternating D/G optimization:
   - **D step**: minimize `BCE(D(real), 1) + BCE(D(G(real)), 0)`
   - **G step**: minimize `λ_adv * BCE(D(G(real)), 1) + λ_rec * MSE(real, G(real))`
3. **Export** — Save Generator to `outputs/gan/generator.pth` and Discriminator to `outputs/gan/discriminator.pth`.

### Evaluation (`src/evaluate.py`)

A single unified evaluation script supports both models via `--model` flag:

1. **Load Model** — Load the chosen model's weights (Autoencoder or Generator).
2. **Collect Test Images** — Gather test/good (label=0) and test/anomaly (label=1) images.
3. **Compute Metrics** — MSE, MAE, and SSIM between original and reconstruction.
4. **Global Metrics** — AUROC and Average Precision using MSE and 1−SSIM.
5. **Per-Category Metrics** — AUROC and AP per category.
6. **Optimal Threshold** — Youden's J statistic on the ROC curve.
7. **Visualizations** — ROC/PR curves, error distributions, AUROC bar chart, reconstruction samples.
8. **Export** — CSV, JSON, and PNG files to `outputs/<model>/evaluation/`.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA (optional, recommended for faster training)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd anomaly_detection_industrial_images

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install torch torchvision pandas pillow tabulate tqdm matplotlib scikit-learn scipy
```

### Dataset

The raw images are **not tracked in git** (too heavy). Only `combined_dataset.zip` is included in the repository. Extract it before training:

```bash
# Windows (PowerShell)
Expand-Archive combined_dataset.zip -DestinationPath .

# Linux / macOS
unzip combined_dataset.zip
```

This creates the `combined_dataset/` folder at the project root with the directory layout described above.

---

## Usage

### Train the Autoencoder

```bash
python src/models/autoencoder/train.py
```

Saves model to `outputs/autoencoder/model.pth`.

### Train the GAN

```bash
python src/models/gan/train.py
```

Saves Generator to `outputs/gan/generator.pth` and Discriminator to `outputs/gan/discriminator.pth`.

### Evaluate a Model

```bash
# Evaluate the Autoencoder (default)
python src/evaluate.py --model autoencoder

# Evaluate the GAN
python src/evaluate.py --model gan
```

Results (plots, CSV, JSON) are saved to `outputs/<model>/evaluation/`.

### Configuration

Shared hyperparameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_HEIGHT` / `IMG_WIDTH` | 256 | Input image resolution |
| `BATCH_SIZE` | 32 | Training and evaluation batch size |
| `NUM_EPOCHS` | 30 | Number of training epochs |
| `LEARNING_RATE` | 1e-3 | Autoencoder Adam learning rate |
| `NUM_IMAGES_TO_MOVE` | 100 | Images moved for test split balancing |

GAN-specific hyperparameters are in `src/models/gan/train.py`:

| Parameter | Default | Description |
|---|---|---|
| `LAMBDA_ADV` | 1.0 | Adversarial loss weight |
| `LAMBDA_REC` | 50.0 | Reconstruction loss weight |
| `LR_G` / `LR_D` | 1e-4 | Generator / Discriminator learning rate |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error between original and reconstructed image (pixel-level) |
| **MAE** | Mean Absolute Error between original and reconstructed image |
| **SSIM** | Structural Similarity Index — measures perceptual similarity |
| **AUROC** | Area Under the ROC Curve — measures binary classification quality (good vs anomaly) |
| **Average Precision** | Area under the Precision-Recall curve — robust to class imbalance |
| **Youden's J** | Optimal threshold for MSE-based classification (`argmax(TPR − FPR)`) |

---

## Results

### Generated Visualizations (per model)

Each model's evaluation outputs are stored in `outputs/<model>/evaluation/`:

| File | Description |
|---|---|
| `roc_pr_curves.png` | ROC and Precision-Recall curves (MSE and SSIM scorers) |
| `error_distribution.png` | Histogram of MSE for good vs anomaly with optimal threshold |
| `auroc_per_category.png` | Bar chart of AUROC per category (color-coded) |
| `reconstructions_good.png` | Original → Reconstruction → Error Map for good samples |
| `reconstructions_anomaly.png` | Original → Reconstruction → Error Map for top anomaly samples |
| `evaluation_results.csv` | Per-image metrics |
| `metrics_per_category.csv` | AUROC and AP per category |
| `evaluation_results.json` | Full results in JSON format |

### Sample Category Performance (Autoencoder, 30 epochs)

Some categories are inherently easier for autoencoders (uniform textures like hazelnut, screw) while others with high intra-class variability (macaroni, PCBs) are more challenging. Train the GAN with `python src/models/gan/train.py` and compare results with `python src/evaluate.py --model gan`.

---

## Known Limitations & Future Work

### Current Limitations

1. **Single generic model across 27 categories** — Training one model on all categories dilutes specialization. Per-category models would likely achieve much higher AUROC.
2. **CPU training** — Without GPU, training takes ~13 min/epoch (AE) or longer (GAN). More epochs and hyperparameter tuning would improve results.
3. **Simple architecture** — The current encoder-decoder is relatively shallow (4+4 layers). Deeper or skip-connected architectures could capture finer details.

### Potential Improvements

- **Per-category models** — Train separate AE/GAN per category for specialized detection.
- **More epochs** — Extend training to 50–100 epochs.
- **Advanced architectures** — VAE, U-Net with skip connections, or memory-augmented autoencoders.
- **Perceptual loss** — Use a pre-trained VGG feature extractor instead of pixel MSE.
- **Data augmentation** — Random rotations, flips, and color jitter.
- **Anomaly localization** — Use per-pixel error maps to highlight anomalous regions.
- **State-of-the-art** — PatchCore, STPM, or EfficientAD for production-grade detection.
- **GANomaly / f-AnoGAN** — More advanced GAN-based approaches with latent-space anomaly scoring.

---

## License

This project is for educational and research purposes. The MVTec AD and VisA datasets have their own respective licenses — please refer to the original sources for terms of use.

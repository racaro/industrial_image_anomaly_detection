# Convolutional Autoencoder

Symmetric encoder-decoder for unsupervised anomaly detection. Trained only on defect-free images; anomalies are detected via high reconstruction error at inference time.

## Architecture

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

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `padding=1` on all layers | Prevents border artifacts that occur with `padding=0` |
| Strided convolutions (`stride=2`) | Learnable downsampling (replaces pooling) |
| BatchNorm after every conv | Stabilizes and accelerates training |
| Sigmoid output | Keeps reconstructions in `[0, 1]` matching the input range |

## Training

- **Loss**: `MSELoss` (pixel-wise reconstruction)
- **Optimizer**: Adam (`lr=1e-3`)
- **Epochs**: 30 (configurable in `src/config.py`)
- **Batch size**: 32
- **Input = Target**: the model learns to reconstruct its own input

```bash
python src/models/autoencoder/train.py
```

Weights are saved to `outputs/autoencoder/model.pth`.

## Anomaly Scoring

At test time, the anomaly score for an image $x$ is:

$$\text{score}(x) = \text{MSE}(x,\; f(x))$$

where $f$ is the trained autoencoder. Higher score → more likely anomalous.

## Evaluation

```bash
python src/evaluate.py --model autoencoder
```

Results (AUROC, AP, visualizations) are saved to `outputs/autoencoder/evaluation/`.

## References

- Bergmann, P. et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
- An, J. & Cho, S. *Variational Autoencoder based Anomaly Detection using Reconstruction Probability.* 2015.

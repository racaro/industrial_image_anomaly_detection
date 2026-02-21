# Reconstruction-based GAN

Adversarially-trained encoder-decoder for unsupervised anomaly detection. The Generator has the same architecture as the Autoencoder so that reconstruction-based anomaly scores are directly comparable.

## Architecture

### Generator (Encoder-Decoder)

Identical to the [Autoencoder](../autoencoder/README.md): 4-layer encoder + 4-layer decoder with `Conv2d(k=4, s=2, p=1)` + BatchNorm + ReLU, and Sigmoid output.

```
Input (3 × 256 × 256) → Encoder → Latent (256 × 16 × 16) → Decoder → Output (3 × 256 × 256)
```

### Discriminator (PatchGAN)

Classifies whether an image is real or reconstructed.

```
Input (3 × 256 × 256)
        │
        ▼
┌─── FEATURES ──────────────────────────────────────┐
│  SpectralNorm(Conv2d(3→64,   k=4, s=2, p=1))      │
│  + LeakyReLU(0.2)                                  │
│  SpectralNorm(Conv2d(64→128,  k=4, s=2, p=1))      │
│  + BN + LeakyReLU(0.2)                             │
│  SpectralNorm(Conv2d(128→256, k=4, s=2, p=1))      │
│  + BN + LeakyReLU(0.2)                             │
│  SpectralNorm(Conv2d(256→512, k=4, s=2, p=1))      │
│  + BN + LeakyReLU(0.2)                             │
└────────────────────────────────────────────────────┘
        │
        ▼
┌─── CLASSIFIER ─────────────────────────────┐
│  AdaptiveAvgPool2d(1) → Flatten            │
│  Linear(512, 1) → Sigmoid                  │
└────────────────────────────────────────────┘
        │
        ▼
Output (B, 1) — probability of being real
```

## Training Strategy

```
                  ┌──────────────┐
  Input ────────► │  Generator   │ ────► Reconstruction
  (real)          │  (G)         │       (fake)
                  └──────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  ┌──────────────┐                ┌──────────────┐
  │ Discriminator │                │   MSE Loss   │
  │     (D)      │                │  (pixel-wise) │
  └──────┬───────┘                └──────┬───────┘
         │                               │
         ▼                               ▼
    L_adv (BCE)                     L_rec (MSE)
```

**Discriminator loss:**

$$L_D = \frac{1}{2}\left[\text{BCE}(D(x),\, 1) + \text{BCE}(D(G(x)),\, 0)\right]$$

**Generator loss:**

$$L_G = \lambda_{\text{adv}} \cdot \text{BCE}(D(G(x)),\, 1) \;+\; \lambda_{\text{rec}} \cdot \text{MSE}(x,\, G(x))$$

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `λ_adv` | 1.0 | Adversarial loss weight |
| `λ_rec` | 50.0 | Reconstruction loss weight (high → prioritize reconstruction) |
| `lr_G` | 1e-4 | Generator learning rate |
| `lr_D` | 1e-4 | Discriminator learning rate |
| `betas` | (0.5, 0.999) | Adam momentum parameters |
| Weight init | DCGAN | `N(0, 0.02)` for conv, `N(1, 0.02)` for BN |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Generator = Autoencoder architecture | Fair comparison — differences come from training, not capacity |
| Spectral normalization on D | Stabilizes discriminator training, prevents mode collapse |
| High `λ_rec` (50) | Prioritizes accurate reconstruction; adversarial term adds sharpness |
| PatchGAN-style D | Focuses on local texture quality rather than global judgement |

## Training

```bash
python src/models/gan/train.py
```

Weights are saved to:
- Generator: `outputs/gan/generator.pth`
- Discriminator: `outputs/gan/discriminator.pth`

## Anomaly Scoring

Same as the Autoencoder — enables direct comparison:

$$\text{score}(x) = \text{MSE}(x,\; G(x))$$

## Evaluation

```bash
python src/evaluate.py --model gan
```

Results (AUROC, AP, visualizations) are saved to `outputs/gan/evaluation/`.

## References

- Schlegl, T. et al. *f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks.* Medical Image Analysis, 2019.
- Akcay, S. et al. *GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.* ACCV 2018.
- Isola, P. et al. *Image-to-Image Translation with Conditional Adversarial Networks (pix2pix).* CVPR 2017.
- Miyato, T. et al. *Spectral Normalization for Generative Adversarial Networks.* ICLR 2018.

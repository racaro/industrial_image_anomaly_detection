# Denoising Diffusion Probabilistic Model (DDPM)

Reconstruction-based anomaly detection using a U-Net denoising backbone with sinusoidal timestep conditioning. Trained only on defect-free images; anomalies yield higher reconstruction error after partial noising + reverse diffusion.

## Architecture

### U-Net Backbone

```
Input (3 × 256 × 256) + Timestep t
        │                    │
        │           SinusoidalEmbedding(128)
        │                    │
        │              MLP(128 → 128)
        │                    │
        ▼                    ▼
┌─── ENCODER (DownBlocks) ──────────────────────────┐
│  DownBlock(3→32,   k=4, s=2) + TimeEmb            │  → 32 × 128 × 128  (s1)
│  DownBlock(32→64,  k=4, s=2) + TimeEmb            │  → 64 × 64 × 64    (s2)
│  DownBlock(64→128, k=4, s=2) + TimeEmb            │  → 128 × 32 × 32   (s3)
│  DownBlock(128→128,k=4, s=2) + TimeEmb            │  → 128 × 16 × 16   (s4)
└────────────────────────────────────────────────────┘
        │
        ▼
┌─── BOTTLENECK ────────────────────────────────────┐
│  ConvBlock(128→128) + TimeEmb                      │  128 × 16 × 16
└────────────────────────────────────────────────────┘
        │
        ▼
┌─── DECODER (UpBlocks + Skip Connections) ─────────┐
│  UpBlock(128, skip=128, →128) + TimeEmb            │  → 128 × 32 × 32
│  UpBlock(128, skip=128, →64)  + TimeEmb            │  → 64 × 64 × 64
│  UpBlock(64,  skip=64,  →32)  + TimeEmb            │  → 32 × 128 × 128
│  UpBlock(32,  skip=32,  →32)  + TimeEmb            │  → 32 × 256 × 256
└────────────────────────────────────────────────────┘
        │
        ▼
    Conv2d(32 → 3, k=1)
        │
        ▼
Output ε̂ (3 × 256 × 256) — predicted noise
```

### Building Blocks

| Block | Composition |
|---|---|
| **ConvBlock** | Conv3×3 → BN → ReLU → TimeEmb addition → Conv3×3 → BN → ReLU |
| **DownBlock** | ConvBlock + Conv4×4(stride=2) downsampling |
| **UpBlock** | ConvTranspose4×4(stride=2) upsample → concat skip → ConvBlock |
| **TimeEmb** | Sinusoidal embedding → Linear(dim→dim) → ReLU |

### Noise Schedules

| Schedule | Formula | Notes |
|---|---|---|
| **Linear** | $\beta_t = \beta_\text{start} + t \cdot (\beta_\text{end} - \beta_\text{start}) / T$ | Simple, but noisy at high $t$ |
| **Cosine** (default) | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Smoother, better for low-res images |

## Training

The model is trained with the **ε-prediction** formulation:

1. Sample clean image $x_0$ from train/good.
2. Sample random timestep $t \sim \text{Uniform}(0, T)$.
3. Sample noise $\varepsilon \sim \mathcal{N}(0, I)$.
4. Create noisy image $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon$.
5. Predict noise $\hat{\varepsilon} = \text{UNet}(x_t, t)$.
6. Minimise $\text{MSE}(\varepsilon, \hat{\varepsilon})$.

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `timesteps` | 1000 | Total diffusion timesteps $T$ |
| `schedule` | cosine | Noise schedule type |
| `epochs` | 40 | Training epochs |
| `lr` | 2e-4 | AdamW learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `scheduler` | CosineAnnealingLR | LR decay (eta_min=1e-6) |
| `batch_size` | 32 | From global config |
| Parameters | ~2.7M | Total trainable parameters |

```bash
python -m src.models.diffusion.train
```

Weights are saved to `outputs/diffusion/model.pth`.

## Anomaly Detection (Inference)

At test time, uses **partial noising + reverse diffusion** (AnoDDPM-inspired):

1. Add noise to input $x_0$ up to timestep $t_\text{start}$ (partial corruption).
2. Run the reverse process from $t_\text{start} \to 0$ to reconstruct.
3. Normal images are well-reconstructed; anomalies are not.

$$\text{score}(x) = \text{MSE}(x,\; \text{reconstruct}(x))$$

| Inference Parameter | Value | Description |
|---|---|---|
| `noise_level` | 0.25 | Fraction of $T$ used for noising ($t_\text{start} = 250$) |
| `inference_steps` | 20 | Steps in the reverse process (strided) |

## Evaluation

```bash
python -m src.evaluate --model diffusion
```

Results (AUROC, AP, visualizations) are saved to `outputs/diffusion/evaluation/`.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Cosine schedule (default) | Smoother noise progression, better for 256×256 images (Nichol & Dhariwal, 2021) |
| ε-prediction (not $x_0$) | Standard DDPM formulation; more stable training |
| Partial noising at inference | Full noising destroys all signal; partial preserves structure for comparison |
| 20 inference steps (strided) | Balance between reconstruction quality and speed |
| `base_channels=32` | Keeps model small (~2.7M) for fair comparison with other approaches |

## References

- Ho, J. et al. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Nichol, A. & Dhariwal, P. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Wyatt, J. et al. *AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models.* CVPR-W 2022.
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017 (sinusoidal embeddings).

# PatchCore Anomaly Detection

Feature-based anomaly detection using pre-trained WideResNet-50 representations and nearest-neighbor scoring. No model training required — builds a memory bank of normal patch features per category and detects anomalies by distance in feature space.

## How It Works

```
┌─── MEMORY BANK BUILDING (offline, per category) ────────────┐
│                                                               │
│  train/good images → WideResNet-50 → patch features           │
│                          ↓                                    │
│                  Coreset subsampling (10%)                     │
│                          ↓                                    │
│               Memory bank M_c  (N × 1536)                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌─── ANOMALY SCORING (inference) ──────────────────────────────┐
│                                                               │
│  test image → WideResNet-50 → patch features (H×W patches)    │
│                          ↓                                    │
│      For each patch p_i:  d_i = min‖p_i − m‖₂  (m ∈ M_c)     │
│                          ↓                                    │
│      Image score = max(d_1, d_2, ..., d_{H×W})                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Feature Extraction

### Baseline: `PatchCoreFeatureExtractor`

Uses WideResNet-50 (pre-trained on ImageNet, all parameters frozen):

| Layer | Output | Channels | Stride |
|---|---|---|---|
| Layer 2 | Feature map | 512 | 8 |
| Layer 3 | Feature map | 1024 | 16 → upsampled to 8 |
| **Concatenated** | **Patch descriptors** | **1536** | **8** |

Input: `(B, 3, 256, 256)` → Output: `(B, 1536, 32, 32)` = 1,024 patches per image.

### Enhanced: `EnhancedFeatureExtractor`

Four evidence-backed improvements for weak categories:

| Improvement | Description | Impact |
|---|---|---|
| **Layer 1 inclusion** | Adds 256-ch features (stride 4) for finer spatial detail | +256 dims → 1792 total |
| **Neighborhood aggregation** | 3×3 average pooling over spatial neighbors (PatchCore paper) | Robust to small shifts |
| **L2 normalization** | Unit-length features reduce scale imbalance between layers | Better distance metrics |
| **Higher resolution** | 512×512 input → 64×64 grid (4× more patches) | Captures fine-grained defects |

## Memory Bank

### Building

```bash
# Build memory banks for all 27 categories
python -m src.models.patchcore.build_memory_bank

# With custom batch size
python -m src.models.patchcore.build_memory_bank --batch-size 8
```

### Coreset Subsampling

Two-stage process to reduce memory bank size while preserving representative coverage:

1. **Random pre-selection**: If $N > 10 \times \text{target}$, randomly sample down.
2. **Greedy farthest-point sampling**: Iteratively select the point farthest from the current coreset.

| Parameter | Value |
|---|---|
| `coreset_ratio` | 0.10 (keep 10% of patches) |
| `max_samples` | 5,000 per category (baseline) / 10,000 (enhanced) |

Memory banks are saved to `outputs/patchcore/<category>/memory_bank.pt`.

## Anomaly Scoring

For a test image with $P$ patches and a memory bank $M$ of size $N$:

$$\text{score}(x) = \max_{i=1}^{P} \min_{m \in M} \| p_i - m \|_2$$

The image-level score is the **maximum nearest-neighbor distance** across all patches — a single highly anomalous patch is sufficient to flag the image.

## Enhanced PatchCore Pipeline

For categories where baseline PatchCore underperforms:

```bash
# Run enhanced pipeline on weak categories
python -m src.models.patchcore.enhanced_features --categories grid screw capsules

# Configuration sweep (tests 4 configs per category)
python -m src.models.patchcore.enhanced_features --sweep --categories grid screw capsules

# Apply best configs from sweep
python -m src.models.patchcore.apply_best_configs
```

### Sweep Configurations

| Config | Resolution | Layer 1 | Neighborhood | L2 Norm | Top-k |
|---|---|---|---|---|---|
| **A** | 512 | Yes | 3×3 | No | 1 |
| **B** | 512 | No | 3×3 | No | 1 |
| **C** | 256 | Yes | 3×3 | Yes | 3 |
| **D** | 512 | Yes | 3×3 | Yes | 1 |

## Evaluation

```bash
# Evaluate PatchCore on all categories
python -m src.evaluate_patchcore

# Compare with another approach
python -m src.evaluate_patchcore --compare-with autoencoder
```

Results (AUROC, AP, per-category breakdown) are saved to `outputs/patchcore/evaluation/`.

## Module Structure

| File | Description |
|---|---|
| `build_memory_bank.py` | Baseline feature extraction, coreset subsampling, memory bank building |
| `enhanced_features.py` | Enhanced extractor, sweep pipeline, per-category evaluation |
| `apply_best_configs.py` | Applies winning sweep configurations to weak categories |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| WideResNet-50 backbone | Strong ImageNet features; standard choice in PatchCore literature |
| Layers 2 & 3 (baseline) | Mid-level features capture both texture and semantic information |
| Coreset subsampling | Reduces memory/compute cost with minimal accuracy loss |
| Max over patches | Single anomalous patch is sufficient for image-level detection |
| Per-category memory banks | Each product type has distinct normal appearance — shared banks fail |
| Reservoir sampling (enhanced) | Prevents OOM when using 512×512 resolution (4× more patches) |

## References

- Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
- Defard, T. et al. *PaDiM: A Patch Distribution Modeling Framework.* ICPR 2021.
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR 2016 (WideResNet backbone).

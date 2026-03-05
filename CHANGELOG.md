# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.8.3] - 2026-03-06

### Added
- **`docs/USAGE.md`**: Full command reference, pipeline overview, configuration tables, evaluation metrics — extracted from README.
- **`docs/RESULTS.md`**: Experimental results tables, analysis & discussion, known limitations, future work — extracted from README.
- **`src/models/diffusion/README.md`**: Architecture details, U-Net diagram, noise schedules, inference strategy.
- **`src/models/patchcore/README.md`**: Pipeline description, memory bank building, coreset subsampling, enhanced features.

### Changed
- **README.md restructured** from 599 lines to ~185 lines: compact overview with links to detailed docs and model READMEs.
- **`src/models/autoencoder/README.md`**: Expanded with V2 architecture, per-category training, performance comparison table.
- **`src/models/gan/README.md`**: Fixed commands to use module syntax, added module structure table.
- **Dataset section** now explains that `combined_dataset/` is built by manually merging MVTec AD and VisA (images not included in repo).
- **`.gitignore`**: Added `docs/` to prevent versioning extended documentation.

## [0.8.2] - 2026-03-06

### Changed
- **Eliminated double inference in `evaluate_per_category.py`**: `evaluate_single_category()` now returns raw labels and combined scores in the result dict; `evaluate_all()` collects them from the first pass instead of re-loading every model and re-inferring all test images a second time (~50% GPU computation saved).

### Fixed
- **Matplotlib memory leak in `evaluate.py`**: Added `plt.close("all")` after reconstruction visualization loop to release figure memory.

### Removed
- **Decorative separator comments** (`# ────`) across all source files (19 modules).
- **Instructional step-by-step markers** (`# --- Step Name ---`, `# ── Section ──`) from training scripts.
- **Usage/Output blocks** from module docstrings — kept only architectural and algorithmic documentation.

## [0.8.1] - 2026-03-06

### Changed
- **GAN Generator reuses Autoencoder**: `Generator` in `src/models/gan/model.py` is now an alias for `Autoencoder` instead of a duplicated 40-line class with identical architecture.
- **Centralized `NUM_WORKERS`**: Extracted `num_workers = 0 if os.name == "nt" else 4` (repeated 13× across the codebase) into `src/config.NUM_WORKERS`.
- **Centralized `compute_combined_score`**: The weighted score fusion `0.3×MSE + 0.3×SSIM + 0.4×Perceptual` (duplicated 4× with inline normalization) is now `src/metrics.compute_combined_score()`.
- **Centralized `prepare_training_data`**: The ~40-line dataset preparation boilerplate (discovery → counting → balancing → validation → DataLoader) duplicated across `autoencoder/train.py` and `gan/train.py` is now `src/dataset.prepare_training_data()`.
- **Updated `src/models/__init__.py`**: Now exports `AutoencoderV2` and `DiffusionModel` alongside existing models.

### Removed
- **`compute_perceptual_loss`** from `src/feature_extractor.py` — unused function (never called anywhere in the codebase).
- **`FancyBboxPatch`** unused import from `src/compare_models.py`.
- **Unused `transforms`** import and dead `transform` variable from `src/models/patchcore/enhanced_features.py`.
- **Unused `pandas`** import from `src/models/gan/train.py` after dataset prep extraction.

## [0.8.0] - 2026-03-05

### Added
- **Enhanced PatchCore for Weak Categories** (`src/models/patchcore/enhanced_features.py`):
  - Four evidence-backed improvements: local neighborhood aggregation (3×3 avg pooling), L2 feature normalization, optional layer 1 inclusion (256ch), higher resolution support (512×512).
  - Memory-efficient reservoir sampling (50K cap) to handle 512×512 resolution without OOM.
  - Configuration sweep mode (`--sweep`) tests 4 configurations per category automatically.
  - `EnhancedConfig` dataclass for clean hyperparameter management.
- **Apply Best Configs** (`src/models/patchcore/apply_best_configs.py`): Applies winning sweep configs to weak categories.

### Results — Weak Category Improvements
- **grid**: 0.5437 → **0.7738** (+0.2301, +42.3%) — Config B: 512 resolution + neighborhood, no L2.
- **capsules**: 0.7858 → **0.8545** (+0.0687, +8.7%) — Config B: 512 resolution + neighborhood, no L2.
- **screw**: 0.6390 → **0.6820** (+0.0430, +6.7%) — Config D: 512 + layer1 + L2 + neighborhood.
- **New Mean AUROC**: 0.8975 → **0.9102** (crosses the 0.91 threshold).

### Key Insights
- L2 normalization destroys texture magnitude information critical for grid-like patterns.
- Higher resolution (512×512 vs 256×256) consistently improves all weak categories.
- Layer 1 features only help when combined with L2 normalization (screw).
- No single config is best for all categories — per-category tuning is essential.

## [0.7.0] - 2026-03-05

### Added
- **Anomaly Localization** (`src/localization.py`): Pixel-level heatmaps using PatchCore patch-distance maps.
  - Per-patch k-NN distance → spatial map → bilinear upsampling to 256×256 → Gaussian smoothing (σ=4).
  - Three visualization modes: standalone heatmap (inferno colormap), overlay on original, and 3-panel grid.
  - Category summary figure: top-5 most anomalous + top-5 most normal images with overlays.
  - Smart grid selection: saves most/least anomalous images (mix of good + anomaly labels).
  - CLI: `python -m src.localization [--categories ...] [--only-anomalies] [--max-per-category N] [--sigma S]`.
  - Output: `outputs/patchcore/localization/{category}/` — 27 categories processed in ~6 min.

## [0.6.0] - 2026-03-05

### Added
- **Per-Category AE Training** (`src/models/autoencoder/train_per_category.py`): Trains 27 independent AE V1 models (one per category), 30 epochs each. CLI with `--categories`, `--epochs`, `--lr`, `--batch-size`. All 27 models trained in 78.9 min on GPU.
- **Per-Category Evaluation** (`src/evaluate_per_category.py`): Evaluates per-category models with VGG perceptual scoring, generates comparison charts vs global model. Mean AUROC: 0.7012 (20/27 categories improved).
- **PatchCore Implementation** (`src/models/patchcore/`):
  - `build_memory_bank.py`: WideResNet-50 backbone (24.9M params, frozen), multi-scale patch features (layers 2+3 → 1536-dim), two-stage GPU-accelerated coreset subsampling (random pre-selection + greedy farthest-point). 27 memory banks built in 12.7 min.
  - `__init__.py`: Exports `PatchCoreFeatureExtractor`, `PATCHCORE_OUTPUT_DIR`.
- **PatchCore Evaluation** (`src/evaluate_patchcore.py`): k-NN scoring (top_k=3, image score = max patch distance), per-category + global metrics, comparison charts. Mean AUROC: **0.8975**.
- **3-Way Comparison** (`src/compare_all_approaches.py`): Side-by-side charts (grouped bars, summary, heatmap) across Global AE V1 vs Per-Category AE vs PatchCore.
- **Project Journal** (`docs/PROJECT_JOURNAL.md`): Comprehensive 16-section documentation covering the entire project journey, technical decisions, problems/solutions, and lessons learned.

### Results
- **PatchCore** (best): Mean AUROC 0.8975, Median 0.9515, 24/27 categories ≥ 0.80, 16/27 ≥ 0.90.
  - Perfect scores: leather (1.00), metal_nut (1.00).
  - Strong performers: zipper (0.99), hazelnut (0.99), carpet (0.98), bottle (0.98).
- **Per-Category AE**: Mean AUROC 0.7012, 9/27 ≥ 0.80, 5/27 ≥ 0.90. Top: cashew (0.97), wood (0.94), candle (0.92).
- **Global AE V1**: Mean AUROC 0.6274, 5/27 ≥ 0.80, 4/27 ≥ 0.90. Limited by single-model-for-all-categories approach.
- **Key finding**: Pre-trained features (PatchCore) outperform trained-from-scratch autoencoders by +27% mean AUROC, with zero training required.

### Technical
- Optimized coreset subsampling: 2-stage approach (random pre-selection 214K→20K + GPU-accelerated greedy farthest-point) reduced build time from 2h+/category to ~15s/category.
- Eliminated double-scoring in PatchCore evaluation pipeline (cached raw scores from per-category eval).

## [0.5.0] - 2026-03-04

### Added
- **Autoencoder V2** (`src/models/autoencoder/model_v2.py`): 5-layer encoder/decoder, 128×8×8 bottleneck (24:1 compression, 2.4M params), LeakyReLU, Dropout2d, Kaiming initialization.
- **AE V2 Training** (`src/models/autoencoder/train_v2.py`): Data augmentation (flips, rotation, color jitter, affine), combined MSE+SSIM loss, AdamW optimizer, cosine annealing LR, gradient clipping.
- **Diffusion DDPM** (`src/models/diffusion/model.py`): UNet with residual blocks, GroupNorm, multi-head attention, 1000 timesteps, cosine β schedule, 2.7M parameters.
- **Diffusion Training** (`src/models/diffusion/train.py`): 40 epochs, AdamW, cosine LR, best-model checkpointing, reconstruction visualizations every 10 epochs.
- **VGG-16 Perceptual Scoring** (`src/feature_extractor.py`): Frozen VGG-16 feature extractor (layers relu1_2, relu2_2, relu3_3, relu4_3), L2 feature distance anomaly scoring.
- **Combined Scoring Pipeline**: Multi-metric fusion `0.3×MSE + 0.3×SSIM + 0.4×Perceptual` with min-max normalization.
- **Model Comparison** (`src/compare_models.py`): 9 comparison charts across all 4 models — global metrics, AUROC heatmap, per-category chart, radar chart, confusion matrices, distribution plots, summary table with weighted ranking.
- CUDA/GPU auto-detection in `src/config.py`.

### Changed
- `src/evaluate.py`: Extended `MODEL_REGISTRY` to support 4 models (autoencoder, autoencoder_v2, gan, diffusion). Added perceptual scoring, combined score, new ROC curves for all 4 scoring methods, per-category metrics include `auroc_perceptual` and `auroc_combined`.
- `src/config.py`: `NUM_EPOCHS` increased from 30 to 50. `DEVICE` now auto-detects CUDA.
- `src/models/autoencoder/__init__.py`: Exports `AutoencoderV2`.
- All models retrained on GPU (NVIDIA RTX 4050, 6GB VRAM, CUDA 12.4) with full 12,050-image dataset.
- `README.md`: Comprehensive rewrite with 4-model architecture details, results tables, analysis, and discussion.

### Results
- **Best model**: Autoencoder V1 — AUROC(Combined) 0.5339, 55.1 ranking points.
- **Top categories**: screw (1.00), cashew (0.94), tile (0.93), fryum (0.91).
- **Key finding**: Single-model-for-all-categories approach limits global AUROC to ~0.53 despite excellent per-category results. Per-category models recommended for production use.

## [0.4.0] - 2026-02-23

### Added
- `tests/` directory with 53 unit and integration tests:
  - `tests/conftest.py` — shared fixtures (dummy tensors, temp dataset dirs)
  - `tests/test_config.py` — paths, device, seed determinism, `ensure_dataset()` logic
  - `tests/test_metrics.py` — SSIM identity, symmetry, range, shape validation
  - `tests/test_models.py` — Autoencoder, Generator, Discriminator forward pass shapes and output ranges
  - `tests/test_dataset.py` — exploration, balancing, path collection, image validation, PyTorch datasets
  - `tests/test_evaluate.py` — model registry, load_model error paths, AUROC/AP with synthetic data

---

## [0.3.0] - 2026-02-23

### Added
- `tox.ini` with environments: `lint`, `format`, `typecheck`, `test`
- `.pre-commit-config.yaml` with ruff lint/format hooks and file hygiene checks
- Per-file `E402` ignores in `pyproject.toml` for entry-point scripts using `sys.path` hacks
- `.tox/` added to `.gitignore`
- `tox>=4.0` added to dev dependencies in `pyproject.toml`

### Changed
- Replaced `Makefile` and `scripts.ps1` with `tox` (Pythonic task runner) + `pre-commit` (git hooks)
- Fixed all 67 ruff lint violations across the codebase:
  - Replaced ambiguous Unicode characters (EN DASH, multiplication sign) with ASCII equivalents
  - Sorted imports with `isort` rules
  - Removed unused imports (`torch`, `ensure_dataset`, `LEARNING_RATE`)
  - Prefixed unused variables with `_` (`_fig`, `_df_validation`)
  - Lowercased constants `C1`/`C2` → `c1`/`c2` in `src/metrics.py` (PEP 8 naming)
  - Added `strict=False` to `zip()` calls
  - Sorted `__all__` in `__init__.py` files
  - Converted `Optional[str]` → `str | None` in `src/logger.py`
- Ran `ruff format` on entire `src/` — 6 files reformatted

### Removed
- `Makefile` (replaced by tox)
- `scripts.ps1` (replaced by tox)

---

## [0.2.0] - 2026-02-23

### Added
- `pyproject.toml` with project metadata, Python version constraint, and optional dev dependencies
- `Makefile` with common commands (install, train, evaluate, test, lint, format, typecheck, clean)
- `scripts.ps1` — PowerShell equivalent of Makefile for Windows users
- Tool configuration for `ruff`, `mypy`, and `pytest` in `pyproject.toml`
- `src/logger.py` — Centralized logging module (console + file handlers, timestamped format)
- `set_seed()` utility in `src/config.py` for full reproducibility (Python, NumPy, PyTorch, cuDNN)
- `SEED = 42` constant in `src/config.py`

### Changed
- Pinned all dependencies in `requirements.txt` to exact versions (`==`) for reproducible builds
- Replaced **all** `print()` calls across the codebase with structured `logging`:
  - `src/config.py` — `ensure_dataset()`
  - `src/dataset.py` — exploration, balancing, validation
  - `src/models/autoencoder/train.py` — training loop and pipeline
  - `src/models/gan/train.py` — training loop and pipeline
  - `src/evaluate.py` — evaluation pipeline, metrics, visualizations
- Training scripts now call `set_seed()` before training for reproducibility

### Environment
- Python 3.12.10, PyTorch 2.10.0, torchvision 0.25.0

---

## [0.1.0] - 2026-02-21

### Added
- Project scaffolding: `src/config.py`, `src/dataset.py`, `src/metrics.py`, `src/evaluate.py`
- Convolutional Autoencoder model (`src/models/autoencoder/`)
- DCGAN-based anomaly detector (`src/models/gan/`)
- Unified evaluation pipeline with AUROC, Average Precision, SSIM, and per-category metrics
- Dataset loader supporting MVTec AD + VisA (27 categories)
- `requirements.txt` with project dependencies
- `.gitignore` excluding dataset, weights, virtual env, and generated outputs

### Removed
- Redundant root-level `evaluate.py` (kept `src/evaluate.py`)

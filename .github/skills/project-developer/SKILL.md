---
name: project-developer
description: Step-by-step guide for evolving the anomaly detection project following senior-level best practices. Covers environment setup, code quality, testing, CI/CD, model improvements, and deployment. Each step requires explicit user approval before execution. Keywords: project, develop, build, improve, next step, roadmap, pipeline, refactor, deploy.
---

# Project Development Skill

A structured, incremental roadmap for taking this project from its current state (v0.1.0) to production-grade quality. **Every step requires explicit user confirmation before execution.**

> **IMPORTANT**: Before starting any step, present the plan to the user and wait for approval. Do NOT execute code, create files, or modify anything until the user says to proceed.

---

## How This Skill Works

1. Read the current project state (CHANGELOG.md, existing files, outputs).
2. Identify which steps have already been completed.
3. Present the **next step** with a clear description of what will change and why.
4. **Wait for user confirmation** before proceeding.
5. After completing each step, update CHANGELOG.md and ask the user to review before moving on.
6. **Commit & push**: Once a step is fully completed and reviewed, create a git commit with a descriptive message (e.g., `feat: step 3 — code quality & linting`) and push to the remote repository. Use [Conventional Commits](https://www.conventionalcommits.org/) format.

> **Git workflow**: Every completed step MUST be committed and pushed before moving to the next one. This ensures incremental progress is saved and visible in the repository history.

---

## Step 1 — Environment & Dependency Hardening

**Goal**: Ensure reproducible, isolated environments with pinned dependencies.

**What will be done**:
- Replace `requirements.txt` with exact pinned versions (`==`) from the current working environment.
- Add a `pyproject.toml` with project metadata, Python version constraint, and optional dev dependencies.
- Add a `Makefile` (or PowerShell equivalents) with common commands: `install`, `train-ae`, `train-gan`, `evaluate`, `test`, `lint`.
- Verify the environment works with a clean install.

**Why**: Unpinned dependencies (`>=`) can break builds when upstream packages release breaking changes. A `pyproject.toml` is the modern Python standard for project metadata.

**Confirmation required**: Present the proposed `pyproject.toml` and pinned `requirements.txt` to the user before writing any files.

---

## Step 2 — Logging & Configuration Improvements

**Goal**: Replace all `print()` calls with structured `logging` and improve configuration management.

**What will be done**:
- Add a `src/logger.py` module with a configured logger (console + file handlers, timestamped format).
- Replace every `print()` in `train.py`, `evaluate.py`, and `dataset.py` with appropriate log levels (`info`, `warning`, `error`, `debug`).
- Add a `src/config.py` enhancement: support loading config from environment variables or a YAML/JSON file for experiment flexibility.
- Add seed-setting utility for full reproducibility (`torch.manual_seed`, `numpy.random.seed`, `torch.backends.cudnn.deterministic`).

**Why**: `print()` is not filterable, not timestamped, and cannot be redirected to files. Structured logging is essential for debugging training runs and production monitoring.

**Confirmation required**: Show the logging format and configuration approach before modifying any existing files.

---

## Step 3 — Code Quality & Linting

**Goal**: Enforce consistent code style and catch bugs early with static analysis.

**What will be done**:
- Add linting config: `ruff` (or `flake8` + `isort` + `black`) with a `pyproject.toml` section.
- Add type checking: `mypy` configuration with strict mode for `src/`.
- Run linters on the entire codebase and fix all issues.
- Add pre-commit hooks (`.pre-commit-config.yaml`) for automatic formatting on commit.

**Why**: Consistent style reduces cognitive load. Type checking catches shape mismatches and None errors before runtime — critical for tensor operations.

**Confirmation required**: Present the linting configuration and list of changes before applying fixes.

---

## Step 4 — Unit & Integration Tests

**Goal**: Add a test suite to catch regressions and validate critical logic.

**What will be done**:
- Create `tests/` directory with proper structure:
  - `tests/test_config.py` — validate paths, device selection, `ensure_dataset()`.
  - `tests/test_dataset.py` — test dataset loading, image validation, label correctness.
  - `tests/test_metrics.py` — test SSIM computation with known inputs (identical images → 1.0, random → <1.0).
  - `tests/test_models.py` — test model forward pass shapes (input 3×256×256 → output 3×256×256), weight loading.
  - `tests/test_evaluate.py` — test metric computation with synthetic data.
- Add `pytest` and `pytest-cov` to dev dependencies.
- Add a `conftest.py` with shared fixtures (dummy tensors, temp directories).

**Why**: Without tests, every change risks breaking existing functionality. Model I/O shape validation is especially important — silent shape mismatches cause subtle bugs.

**Confirmation required**: Present the test plan and fixture design before creating any test files.

---

## Step 5 — CI/CD Pipeline

**Goal**: Automate testing, linting, and quality checks on every push/PR.

**What will be done**:
- Create `.github/workflows/ci.yml`:
  - Trigger on push and pull request to `main`.
  - Matrix: Python 3.10, 3.11, 3.12.
  - Steps: install dependencies → lint (`ruff`) → type check (`mypy`) → run tests (`pytest`) → coverage report.
- Add branch protection rules recommendation for `main`.
- Add status badges to `README.md`.

**Why**: CI catches errors before they reach `main`. Matrix testing ensures compatibility across Python versions.

**Confirmation required**: Present the workflow YAML before committing.

---

## Step 6 — Data Augmentation & Preprocessing Pipeline

**Goal**: Improve model generalization with data augmentation and a cleaner data pipeline.

**What will be done**:
- Add configurable augmentation transforms in `src/dataset.py` or a new `src/transforms.py`:
  - Training: `RandomHorizontalFlip`, `RandomRotation(±10°)`, `ColorJitter(brightness, contrast)`, `RandomResizedCrop`.
  - Evaluation: only `Resize` + `ToTensor` (deterministic).
- Make augmentation configurable via `config.py` (enable/disable, parameters).
- Add dataset statistics computation (mean, std) for proper normalization.

**Why**: The current pipeline only resizes and normalizes. Augmentation increases effective training data and reduces overfitting, especially for categories with few training images.

**Confirmation required**: Present the augmentation strategy and expected impact before modifying the data pipeline.

---

## Step 7 — Training Improvements

**Goal**: Make training more robust, resumable, and observable.

**What will be done**:
- Add **checkpointing**: save model + optimizer state every N epochs. Support `--resume` flag.
- Add **early stopping**: monitor validation loss (using a held-out portion of train/good) and stop if no improvement for P epochs.
- Add **learning rate scheduling**: `ReduceLROnPlateau` or `CosineAnnealingLR`.
- Add **TensorBoard / CSV logging**: log loss per epoch, learning rate, sample reconstructions.
- Add **training timer**: log per-epoch and total training time.
- Refactor shared training logic into a base `Trainer` class (Template Method pattern).

**Why**: Without checkpointing, a crash at epoch 29/30 loses all progress. Early stopping prevents overfitting. LR scheduling improves convergence.

**Confirmation required**: Present the `Trainer` base class design and checkpoint format before implementation.

---

## Step 8 — Model Architecture Improvements

**Goal**: Implement more powerful architectures for better anomaly detection.

**What will be done** (one at a time, user chooses):
- **Option A**: Add skip connections (U-Net style) to the Autoencoder for better detail preservation.
- **Option B**: Implement a Variational Autoencoder (VAE) with KL divergence regularization.
- **Option C**: Add perceptual loss using a pre-trained VGG-16 feature extractor.
- **Option D**: Implement a state-of-the-art method (PatchCore, EfficientAD, or STPM).

Each new model follows the existing pattern: `src/models/<name>/model.py`, `train.py`, `README.md`, `__init__.py`.

**Why**: The current 4-layer encoder-decoder is shallow. Skip connections preserve fine-grained details crucial for anomaly detection. Perceptual loss captures high-level features.

**Confirmation required**: Present the architecture design with diagrams, expected parameter count, and training strategy before implementing.

---

## Step 9 — Per-Category Models & Experiment Tracking

**Goal**: Train specialized models per category and track experiments systematically.

**What will be done**:
- Add `--category` flag to training scripts to train on a single category.
- Add experiment tracking: log hyperparameters, metrics, and model paths per run (MLflow, Weights & Biases, or a simple JSON-based tracker).
- Add comparison utilities: generate tables and plots comparing all models across all categories.
- Add `src/experiments/` module for managing experiment configurations.

**Why**: A single model across 27 diverse categories dilutes performance. Per-category models are standard practice in anomaly detection benchmarks.

**Confirmation required**: Present the experiment tracking design and per-category training strategy before implementation.

---

## Step 10 — Anomaly Localization

**Goal**: Generate pixel-level anomaly maps to highlight defective regions.

**What will be done**:
- Compute per-pixel error maps (absolute reconstruction error).
- Apply Gaussian smoothing to error maps for cleaner visualization.
- Add heatmap overlay on original images.
- Compute pixel-level AUROC (if ground-truth masks are available).
- Save localization visualizations to `outputs/<model>/evaluation/localization/`.

**Why**: Image-level classification tells you *if* an image is anomalous; localization tells you *where*. This is critical for real-world industrial inspection.

**Confirmation required**: Present the localization approach and visualization examples before implementing.

---

## Step 11 — API & Inference Pipeline

**Goal**: Create a clean inference API for single-image and batch prediction.

**What will be done**:
- Add `src/inference.py` with:
  - `predict(image_path, model_name, threshold) → {"label", "score", "error_map"}`
  - `predict_batch(image_dir, model_name, threshold) → DataFrame`
- Add CLI entry point: `python -m src.inference --image path/to/img.png --model autoencoder`.
- Add optional REST API with FastAPI (simple `/predict` endpoint).
- Add model export to ONNX for deployment.

**Why**: Training and evaluation scripts exist, but there is no clean way to run inference on new images. An API is the bridge to production.

**Confirmation required**: Present the API design and endpoint specification before implementing.

---

## Step 12 — Documentation & Packaging

**Goal**: Polish documentation and make the project installable/publishable.

**What will be done**:
- Update `README.md` with latest results, new models, and updated instructions.
- Add `CONTRIBUTING.md` with development setup and code style guidelines.
- Add architecture decision records (`docs/adr/`) for key design choices.
- Ensure `pyproject.toml` supports `pip install -e .`.
- Add `LICENSE` file.
- Update `CHANGELOG.md` to reflect the new version.
- Tag the release in git.

**Why**: Good documentation lowers the barrier for contributors and demonstrates professional engineering. A proper package structure enables reuse.

**Confirmation required**: Present the updated README outline and package structure before finalizing.

---

## Progress Tracking

After each step, update `CHANGELOG.md` with the changes under a new version entry. Use semantic versioning:

| Completed Steps | Suggested Version |
|---|---|
| Steps 1–3 | 0.2.0 (Developer Experience) |
| Steps 4–5 | 0.3.0 (Quality Assurance) |
| Steps 6–7 | 0.4.0 (Training Pipeline) |
| Step 8 | 0.5.0 (New Architecture) |
| Steps 9–10 | 0.6.0 (Advanced Detection) |
| Steps 11–12 | 1.0.0 (Production Ready) |
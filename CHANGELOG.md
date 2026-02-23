# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

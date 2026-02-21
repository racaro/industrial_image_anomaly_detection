# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

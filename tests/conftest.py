"""
Shared pytest fixtures for the anomaly detection test suite.
"""

import os

import pytest
import torch
from PIL import Image

# ──────────────────────────────────────────────
# Tensor fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """Random (B, C, H, W) tensor in [0, 1] matching model input size."""
    return torch.rand(4, 3, 256, 256)


@pytest.fixture
def ones_batch() -> torch.Tensor:
    """Constant tensor — useful for SSIM identity tests."""
    return torch.ones(2, 3, 256, 256) * 0.5


# ──────────────────────────────────────────────
# Fake dataset directory
# ──────────────────────────────────────────────

_CATEGORIES = ["bottle", "cable"]


def _create_dummy_image(path: str, size: tuple[int, int] = (64, 64)) -> None:
    """Write a small solid-colour RGB image to *path*."""
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(path)


@pytest.fixture
def tmp_dataset(tmp_path) -> str:
    """
    Create a minimal dataset directory under tmp_path with the structure
    expected by the project:
        <dataset>/
            <category>/
                train/good/
                test/good/
                test/anomaly/
    Each folder contains 3 tiny dummy images.
    """
    ds_root = str(tmp_path / "combined_dataset")
    for cat in _CATEGORIES:
        for subfolder in ["train/good", "test/good", "test/anomaly"]:
            folder = os.path.join(ds_root, cat, *subfolder.split("/"))
            os.makedirs(folder, exist_ok=True)
            for i in range(3):
                _create_dummy_image(os.path.join(folder, f"img_{i:03d}.png"))
    return ds_root


@pytest.fixture
def tmp_dataset_no_test_good(tmp_path) -> str:
    """
    Dataset where one category has NO test/good images
    (used to test balancing logic).
    """
    ds_root = str(tmp_path / "combined_dataset_unbalanced")
    for cat in _CATEGORIES:
        # train/good always present
        train_dir = os.path.join(ds_root, cat, "train", "good")
        os.makedirs(train_dir, exist_ok=True)
        for i in range(5):
            _create_dummy_image(os.path.join(train_dir, f"img_{i:03d}.png"))

        # test/anomaly always present
        anom_dir = os.path.join(ds_root, cat, "test", "anomaly")
        os.makedirs(anom_dir, exist_ok=True)
        for i in range(3):
            _create_dummy_image(os.path.join(anom_dir, f"img_{i:03d}.png"))

    # Only the first category has test/good — the second does NOT
    good_dir = os.path.join(ds_root, _CATEGORIES[0], "test", "good")
    os.makedirs(good_dir, exist_ok=True)
    for i in range(3):
        _create_dummy_image(os.path.join(good_dir, f"img_{i:03d}.png"))

    return ds_root

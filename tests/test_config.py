"""
Tests for src/config.py — paths, device, seed utility.
"""

import os
import random

import numpy as np
import pytest
import torch


class TestConstants:
    """Verify that critical constants are sane."""

    def test_project_root_exists(self):
        from src.config import PROJECT_ROOT

        assert os.path.isdir(PROJECT_ROOT)

    def test_device_is_torch_device(self):
        from src.config import DEVICE

        assert isinstance(DEVICE, torch.device)

    def test_seed_is_int(self):
        from src.config import SEED

        assert isinstance(SEED, int)

    def test_image_dimensions_positive(self):
        from src.config import IMG_HEIGHT, IMG_WIDTH

        assert IMG_HEIGHT > 0
        assert IMG_WIDTH > 0

    def test_hyperparameters_positive(self):
        from src.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

        assert BATCH_SIZE > 0
        assert NUM_EPOCHS > 0
        assert LEARNING_RATE > 0


class TestSetSeed:
    """Verify that set_seed produces deterministic outputs."""

    def test_torch_determinism(self):
        from src.config import set_seed

        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_numpy_determinism(self):
        from src.config import set_seed

        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_determinism(self):
        from src.config import set_seed

        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestEnsureDataset:
    """Test ensure_dataset() logic without actually extracting."""

    def test_returns_none_when_folder_exists(self, tmp_path, monkeypatch):
        """If dataset dir already exists, ensure_dataset should return immediately."""
        from src import config

        fake_ds = tmp_path / "combined_dataset"
        fake_ds.mkdir()
        monkeypatch.setattr(config, "DATASET_PATH", str(fake_ds))

        # Should not raise
        config.ensure_dataset()

    def test_raises_when_neither_folder_nor_zip(self, tmp_path, monkeypatch):
        """If neither the folder nor zip exist, it should raise FileNotFoundError."""
        from src import config

        monkeypatch.setattr(config, "DATASET_PATH", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(config, "DATASET_ZIP_PATH", str(tmp_path / "nonexistent.zip"))

        with pytest.raises(FileNotFoundError):
            config.ensure_dataset()

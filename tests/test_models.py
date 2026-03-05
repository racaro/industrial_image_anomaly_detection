"""
Tests for model architectures — forward pass shapes, parameter counts, etc.
"""

import pytest
import torch

from src.models.autoencoder import Autoencoder
from src.models.gan import Discriminator, Generator

# ──────────────────────────────────────────────
# Autoencoder
# ──────────────────────────────────────────────


class TestAutoencoder:
    """Validate Autoencoder I/O shapes and properties."""

    @pytest.fixture(autouse=True)
    def _build_model(self):
        self.model = Autoencoder().eval()

    def test_output_shape_matches_input(self, dummy_batch: torch.Tensor):
        """Output (B, 3, 256, 256) should match input shape."""
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.shape == dummy_batch.shape

    def test_output_range_zero_to_one(self, dummy_batch: torch.Tensor):
        """Sigmoid output should be in [0, 1]."""
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_image(self):
        """Should also work with batch size = 1."""
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 3, 256, 256)

    def test_has_encoder_and_decoder(self):
        assert hasattr(self.model, "encoder")
        assert hasattr(self.model, "decoder")

    def test_parameter_count_positive(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 0


# ──────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────


class TestGenerator:
    """Validate Generator I/O shapes (same architecture as AE)."""

    @pytest.fixture(autouse=True)
    def _build_model(self):
        self.model = Generator().eval()

    def test_output_shape_matches_input(self, dummy_batch: torch.Tensor):
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.shape == dummy_batch.shape

    def test_output_range_zero_to_one(self, dummy_batch: torch.Tensor):
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_image(self):
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 3, 256, 256)


# ──────────────────────────────────────────────
# Discriminator
# ──────────────────────────────────────────────


class TestDiscriminator:
    """Validate Discriminator I/O shapes."""

    @pytest.fixture(autouse=True)
    def _build_model(self):
        self.model = Discriminator().eval()

    def test_output_shape(self, dummy_batch: torch.Tensor):
        """Output should be (B, 1) probability."""
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.shape == (dummy_batch.shape[0], 1)

    def test_output_range_zero_to_one(self, dummy_batch: torch.Tensor):
        """Sigmoid output should be in [0, 1]."""
        with torch.no_grad():
            out = self.model(dummy_batch)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_image(self):
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 1)

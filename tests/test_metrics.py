"""
Tests for src/metrics.py — SSIM computation.
"""

import torch

from src.metrics import compute_ssim_batch


class TestComputeSSIMBatch:
    """Validate SSIM behaviour for known edge cases."""

    def test_identical_images_yield_one(self, dummy_batch: torch.Tensor):
        """SSIM(x, x) should be ~1.0 for any x."""
        ssim = compute_ssim_batch(dummy_batch, dummy_batch)
        assert ssim.shape == (dummy_batch.shape[0],)
        assert torch.allclose(ssim, torch.ones_like(ssim), atol=1e-5)

    def test_different_images_below_one(self, dummy_batch: torch.Tensor):
        """SSIM between two independent random batches should be < 1."""
        other = torch.rand_like(dummy_batch)
        ssim = compute_ssim_batch(dummy_batch, other)
        assert (ssim < 1.0).all()

    def test_constant_images(self, ones_batch: torch.Tensor):
        """SSIM between identical constant images should be ~1.0."""
        ssim = compute_ssim_batch(ones_batch, ones_batch)
        assert torch.allclose(ssim, torch.ones_like(ssim), atol=1e-5)

    def test_output_shape_matches_batch(self):
        """Output tensor should have shape (B,) matching input batch size."""
        for batch_size in [1, 4, 8]:
            x = torch.rand(batch_size, 3, 64, 64)
            ssim = compute_ssim_batch(x, x)
            assert ssim.shape == (batch_size,)

    def test_ssim_is_symmetric(self, dummy_batch: torch.Tensor):
        """SSIM(x, y) should equal SSIM(y, x)."""
        other = torch.rand_like(dummy_batch)
        ssim_xy = compute_ssim_batch(dummy_batch, other)
        ssim_yx = compute_ssim_batch(other, dummy_batch)
        assert torch.allclose(ssim_xy, ssim_yx, atol=1e-6)

    def test_ssim_values_in_valid_range(self, dummy_batch: torch.Tensor):
        """SSIM should be in [-1, 1] (typically [0, 1] for non-negative images)."""
        other = torch.rand_like(dummy_batch)
        ssim = compute_ssim_batch(dummy_batch, other)
        assert (ssim >= -1.0).all()
        assert (ssim <= 1.0).all()

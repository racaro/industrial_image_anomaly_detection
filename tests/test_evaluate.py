"""
Tests for src/evaluate.py — model registry, metric computation with synthetic data.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import MODEL_REGISTRY, load_model


class TestModelRegistry:
    """Verify the MODEL_REGISTRY structure."""

    def test_autoencoder_in_registry(self):
        assert "autoencoder" in MODEL_REGISTRY

    def test_gan_in_registry(self):
        assert "gan" in MODEL_REGISTRY

    def test_registry_keys(self):
        for _name, info in MODEL_REGISTRY.items():
            assert "class" in info
            assert "weights" in info
            assert "eval_dir" in info


class TestLoadModel:
    """Test load_model error paths (we don't have trained weights in CI)."""

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent_model")

    def test_missing_weights_raises(self, tmp_path, monkeypatch):
        """If weight file doesn't exist, should raise FileNotFoundError."""
        from src import evaluate

        # Point the registry to a non-existent file
        monkeypatch.setitem(
            evaluate.MODEL_REGISTRY,
            "autoencoder",
            {
                "class": evaluate.MODEL_REGISTRY["autoencoder"]["class"],
                "weights": str(tmp_path / "missing.pth"),
                "eval_dir": str(tmp_path / "eval"),
            },
        )
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            load_model("autoencoder")


class TestMetricComputation:
    """
    Test AUROC / Average Precision with synthetic scores,
    validating the same metric functions used by evaluate.py.
    """

    def test_perfect_scores_give_auroc_one(self):
        """Perfect separation should yield AUROC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert roc_auc_score(y_true, scores) == pytest.approx(1.0)

    def test_random_scores_auroc_around_half(self):
        """Random scores should yield AUROC ~ 0.5 (within tolerance)."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=1000)
        scores = rng.rand(1000)
        auroc = roc_auc_score(y_true, scores)
        assert 0.3 < auroc < 0.7  # generous tolerance

    def test_perfect_ap(self):
        """Perfect separation should yield AP = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert average_precision_score(y_true, scores) == pytest.approx(1.0)

    def test_inverted_scores_give_auroc_zero(self):
        """Completely inverted scores should yield AUROC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert roc_auc_score(y_true, scores) == pytest.approx(0.0)

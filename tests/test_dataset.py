"""
Tests for src/dataset.py — exploration, balancing, validation, PyTorch datasets.
"""

import os

import torch

from src.dataset import (
    AnomalyImageDataset,
    EvalImageDataset,
    balance_test_good,
    build_distribution_df,
    collect_image_paths,
    collect_test_images,
    count_images,
    get_categories,
    validate_images,
)

# ──────────────────────────────────────────────
# Exploration helpers
# ──────────────────────────────────────────────


class TestGetCategories:
    def test_returns_sorted_categories(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        assert cats == ["bottle", "cable"]

    def test_empty_dir(self, tmp_path):
        assert get_categories(str(tmp_path)) == []


class TestCountImages:
    def test_correct_counts(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        counts = count_images(tmp_dataset, cats)
        for cat in cats:
            assert counts[cat]["train_good"] == 3
            assert counts[cat]["test_good"] == 3
            assert counts[cat]["test_anomaly"] == 3


class TestBuildDistributionDf:
    def test_dataframe_columns(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        counts = count_images(tmp_dataset, cats)
        df = build_distribution_df(counts)
        expected_cols = {"Category", "Train Good", "Test Good", "Test Anomaly", "Total Images"}
        assert set(df.columns) == expected_cols

    def test_total_column(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        counts = count_images(tmp_dataset, cats)
        df = build_distribution_df(counts)
        for _, row in df.iterrows():
            assert row["Total Images"] == row["Train Good"] + row["Test Good"] + row["Test Anomaly"]


# ──────────────────────────────────────────────
# Balancing
# ──────────────────────────────────────────────


class TestBalanceTestGood:
    def test_moves_images_to_test_good(self, tmp_dataset_no_test_good: str):
        """The 'cable' category has no test/good — balancing should move some."""
        balance_test_good(tmp_dataset_no_test_good, ["cable"], num_to_move=2)
        test_good_dir = os.path.join(tmp_dataset_no_test_good, "cable", "test", "good")
        assert os.path.isdir(test_good_dir)
        moved = [f for f in os.listdir(test_good_dir) if f.endswith(".png")]
        assert len(moved) == 2


# ──────────────────────────────────────────────
# Path collection
# ──────────────────────────────────────────────


class TestCollectImagePaths:
    def test_collects_train_good(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats, subfolder=os.path.join("train", "good"))
        assert len(data) == 6  # 2 cats * 3 imgs
        assert all("Image Path" in d and "Category" in d for d in data)

    def test_paths_exist(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats)
        for d in data:
            assert os.path.isfile(d["Image Path"])


class TestCollectTestImages:
    def test_collects_test_images(self, tmp_dataset: str):
        records = collect_test_images(tmp_dataset)
        # 2 cats * (3 good + 3 anomaly) = 12
        assert len(records) == 12

    def test_labels_correct(self, tmp_dataset: str):
        records = collect_test_images(tmp_dataset)
        good = [r for r in records if r["label"] == 0]
        anom = [r for r in records if r["label"] == 1]
        assert len(good) == 6
        assert len(anom) == 6


# ──────────────────────────────────────────────
# Image validation
# ──────────────────────────────────────────────


class TestValidateImages:
    def test_all_valid_images_pass(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats)
        valid, df_val = validate_images(data)
        assert len(valid) == len(data)
        assert len(df_val) > 0

    def test_corrupt_images_discarded(self, tmp_path):
        """A file with invalid image content should be discarded."""
        bad_img = tmp_path / "bad.png"
        bad_img.write_text("not-an-image")
        data = [{"Category": "test", "Image Path": str(bad_img)}]
        valid, _ = validate_images(data)
        assert len(valid) == 0


# ──────────────────────────────────────────────
# PyTorch Datasets
# ──────────────────────────────────────────────


class TestAnomalyImageDataset:
    def test_len(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats)
        ds = AnomalyImageDataset(data, img_h=64, img_w=64)
        assert len(ds) == len(data)

    def test_getitem_returns_image_pair(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats)
        ds = AnomalyImageDataset(data, img_h=64, img_w=64)
        img, target = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert torch.equal(img, target)

    def test_values_in_zero_one(self, tmp_dataset: str):
        cats = get_categories(tmp_dataset)
        data = collect_image_paths(tmp_dataset, cats)
        ds = AnomalyImageDataset(data, img_h=64, img_w=64)
        img, _ = ds[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0


class TestEvalImageDataset:
    def test_len(self, tmp_dataset: str):
        records = collect_test_images(tmp_dataset)
        ds = EvalImageDataset(records, img_h=64, img_w=64)
        assert len(ds) == len(records)

    def test_getitem_returns_image_label_idx(self, tmp_dataset: str):
        records = collect_test_images(tmp_dataset)
        ds = EvalImageDataset(records, img_h=64, img_w=64)
        img, label, idx = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert label in (0, 1)
        assert idx == 0

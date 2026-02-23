"""
Dataset utilities: exploration, balancing, validation, and PyTorch datasets.
"""

import os
import shutil
from collections import Counter

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMAGE_EXTENSIONS, IMG_HEIGHT, IMG_WIDTH
from src.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# EXPLORATION
# ──────────────────────────────────────────────


def get_categories(dataset_path: str) -> list[str]:
    """Return sorted list of categories (subdirectories) in the dataset."""
    return sorted(d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)))


def _count_images_in_folder(folder_path: str) -> int:
    """Count image files inside a folder."""
    if not os.path.isdir(folder_path):
        return 0
    return sum(
        1
        for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder_path, f))
    )


def count_images(dataset_path: str, categories: list[str]) -> dict:
    """Count images in train/good, test/good, and test/anomaly for each category."""
    counts: dict[str, dict[str, int]] = {}
    for cat in categories:
        cat_path = os.path.join(dataset_path, cat)
        counts[cat] = {
            "train_good": _count_images_in_folder(os.path.join(cat_path, "train", "good")),
            "test_good": _count_images_in_folder(os.path.join(cat_path, "test", "good")),
            "test_anomaly": _count_images_in_folder(os.path.join(cat_path, "test", "anomaly")),
        }
    return counts


def build_distribution_df(image_counts: dict) -> pd.DataFrame:
    """Build a summary DataFrame of the image distribution."""
    rows = []
    for cat, c in image_counts.items():
        rows.append(
            {
                "Category": cat,
                "Train Good": c["train_good"],
                "Test Good": c["test_good"],
                "Test Anomaly": c["test_anomaly"],
                "Total Images": c["train_good"] + c["test_good"] + c["test_anomaly"],
            }
        )
    df = pd.DataFrame(rows).sort_values("Category").reset_index(drop=True)
    return df


def print_distribution_summary(df: pd.DataFrame, title: str = "Image Distribution") -> None:
    """Log the dataset distribution summary."""
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)
    logger.info("\n%s", df.to_markdown(index=False))
    logger.info("  Total Train Good:  %d", df["Train Good"].sum())
    logger.info("  Total Test Good:   %d", df["Test Good"].sum())
    logger.info("  Total Test Anomaly:%d", df["Test Anomaly"].sum())
    logger.info("  Overall Total:     %d", df["Total Images"].sum())


# ──────────────────────────────────────────────
# BALANCING
# ──────────────────────────────────────────────


def balance_test_good(
    dataset_path: str,
    categories_missing: list[str],
    num_to_move: int = 100,
) -> None:
    """
    Move `num_to_move` images from train/good → test/good
    for categories that have no test/good images.
    """
    logger.info(
        "Moving up to %d images from train/good → test/good for %d categories without test/good...",
        num_to_move,
        len(categories_missing),
    )

    for cat in categories_missing:
        src = os.path.join(dataset_path, cat, "train", "good")
        dst = os.path.join(dataset_path, cat, "test", "good")
        os.makedirs(dst, exist_ok=True)

        if not os.path.isdir(src):
            logger.warning("[%s] No train/good folder - skipped.", cat)
            continue

        image_files = [
            f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(src, f))
        ]
        to_move = image_files[:num_to_move]

        moved = 0
        for fname in to_move:
            try:
                shutil.move(os.path.join(src, fname), os.path.join(dst, fname))
                moved += 1
            except Exception as e:
                logger.error("Error moving %s: %s", fname, e)

        logger.info("[%s] Moved %d/%d images.", cat, moved, len(to_move))

    logger.info("Balancing completed.")


# ──────────────────────────────────────────────
# IMAGE PATH COLLECTION
# ──────────────────────────────────────────────


def collect_image_paths(
    dataset_path: str,
    categories: list[str],
    subfolder: str = os.path.join("train", "good"),
) -> list[dict]:
    """
    Collect full image paths and their category.
    Returns a list of dicts with 'Category' and 'Image Path'.
    """
    image_data: list[dict] = []
    for cat in categories:
        folder = os.path.join(dataset_path, cat, subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            full_path = os.path.join(folder, fname)
            if fname.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(full_path):
                image_data.append({"Category": cat, "Image Path": full_path})
    return image_data


def collect_test_images(dataset_path: str) -> list[dict]:
    """Collect test/good and test/anomaly images from all categories."""
    records = []
    categories = get_categories(dataset_path)
    for cat in categories:
        for subfolder, label in [("test/good", 0), ("test/anomaly", 1)]:
            folder = os.path.join(dataset_path, cat, subfolder.replace("/", os.sep))
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if fname.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(fpath):
                    records.append(
                        {
                            "category": cat,
                            "label": label,
                            "label_name": "good" if label == 0 else "anomaly",
                            "path": fpath,
                        }
                    )
    return records


# ──────────────────────────────────────────────
# IMAGE VALIDATION
# ──────────────────────────────────────────────


def validate_images(
    image_data: list[dict],
) -> tuple[list[dict], pd.DataFrame]:
    """
    Verify integrity and characteristics of all images.
    - Discards corrupt images.
    - Reports distribution of sizes, modes, and formats by category.
    Returns (clean image_data list, summary DataFrame).
    """
    valid: list[dict] = []
    broken: list[dict] = []
    size_counter: Counter = Counter()
    mode_counter: Counter = Counter()
    fmt_counter: Counter = Counter()
    cat_sizes: dict[str, Counter] = {}

    logger.info("Validating image integrity...")
    for entry in image_data:
        fpath = entry["Image Path"]
        cat = entry["Category"]
        try:
            img = Image.open(fpath)
            img.verify()
            img = Image.open(fpath)
            w, h = img.size
            size_counter[(w, h)] += 1
            mode_counter[img.mode] += 1
            fmt_counter[img.format or "unknown"] += 1
            cat_sizes.setdefault(cat, Counter())[(w, h)] += 1
            valid.append(entry)
        except Exception as e:
            broken.append({"Category": cat, "Path": fpath, "Error": str(e)})

    logger.info("Valid images:   %d", len(valid))
    logger.info("Corrupt images: %d", len(broken))
    if broken:
        for b in broken[:10]:
            logger.warning("[%s] %s: %s", b["Category"], b["Path"], b["Error"])

    logger.info("Color modes:    %s", dict(mode_counter))
    logger.info("Formats:        %s", dict(fmt_counter))
    logger.info("Unique sizes:   %d", len(size_counter))
    for (w, h), c in size_counter.most_common():
        logger.info("  %dx%d: %d imgs", w, h, c)

    rows = []
    for cat in sorted(cat_sizes):
        sizes_str = ", ".join(f"{w}x{h}({c})" for (w, h), c in cat_sizes[cat].most_common())
        rows.append({"Category": cat, "Count": sum(cat_sizes[cat].values()), "Sizes": sizes_str})
    df_val = pd.DataFrame(rows)
    logger.info("Detail by category:\n%s", df_val.to_markdown(index=False))

    logger.info(
        "All images will be resized to %dx%d and normalized to [0,1] during training.",
        IMG_HEIGHT,
        IMG_WIDTH,
    )

    return valid, df_val


# ──────────────────────────────────────────────
# PYTORCH DATASETS
# ──────────────────────────────────────────────


class AnomalyImageDataset(Dataset):
    """
    Custom dataset that loads images, resizes them to
    (img_h, img_w), and normalizes to [0, 1] with ToTensor().
    Returns (image, image) for autoencoder / generator training.
    """

    def __init__(self, image_data: list[dict], img_h: int, img_w: int):
        self.image_data = image_data
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int):
        img_path = self.image_data[idx]["Image Path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, image


class EvalImageDataset(Dataset):
    """Loads images with metadata (category, label, path) for evaluation."""

    def __init__(self, records: list[dict], img_h: int, img_w: int):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        image = self.transform(image)
        label = rec["label"]
        return image, label, idx

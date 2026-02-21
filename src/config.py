"""
Shared configuration constants for all training and evaluation scripts.
"""

import os
import torch

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "combined_dataset")
DATASET_ZIP_PATH = os.path.join(PROJECT_ROOT, "combined_dataset.zip")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ──────────────────────────────────────────────
# IMAGE & TRAINING
# ──────────────────────────────────────────────

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_IMAGES_TO_MOVE = 100  # images to move from train/good → test/good when missing

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dataset() -> None:
    """
    Extract combined_dataset.zip into PROJECT_ROOT if the
    combined_dataset/ folder does not already exist.
    """
    if os.path.isdir(DATASET_PATH):
        return

    if not os.path.isfile(DATASET_ZIP_PATH):
        raise FileNotFoundError(
            f"Dataset not found.\n"
            f"  Expected folder: {DATASET_PATH}\n"
            f"  Expected zip:    {DATASET_ZIP_PATH}\n"
            f"Place combined_dataset.zip in the project root."
        )

    import zipfile

    print(f"Extracting {DATASET_ZIP_PATH} → {PROJECT_ROOT} ...")
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zf:
        zf.extractall(PROJECT_ROOT)
    print("Extraction complete.\n")

"""
Shared configuration constants for all training and evaluation scripts.
"""

import os
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "combined_dataset")
DATASET_ZIP_PATH = os.path.join(PROJECT_ROOT, "combined_dataset.zip")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_IMAGES_TO_MOVE = 100  # images to move from train/good → test/good when missing

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

NUM_WORKERS: int = 0 if os.name == "nt" else 4

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for full reproducibility.

    Configures Python, NumPy, and PyTorch (CPU + CUDA) random number
    generators, and enables deterministic cuDNN behaviour.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dataset() -> None:
    """
    Extract combined_dataset.zip into PROJECT_ROOT if the
    combined_dataset/ folder does not already exist.
    """
    import logging

    logger = logging.getLogger(__name__)

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

    logger.info("Extracting %s → %s ...", DATASET_ZIP_PATH, PROJECT_ROOT)
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zf:
        zf.extractall(PROJECT_ROOT)
    logger.info("Extraction complete.")

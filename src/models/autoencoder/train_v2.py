"""
Training pipeline for the improved Autoencoder V2.

Improvements over V1 training:
  - Data augmentation (colour jitter, flips, slight rotation)
  - Combined loss: MSE + SSIM + optional perceptual (VGG)
  - AdamW optimiser + Cosine Annealing LR
  - Gradient clipping
"""

import os
import random
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import (
    DATASET_PATH,
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_WORKERS,
    OUTPUTS_DIR,
    ensure_dataset,
    set_seed,
)
from src.dataset import (
    collect_image_paths,
    get_categories,
    validate_images,
)
from src.logger import get_logger
from src.metrics import compute_ssim_batch
from src.models.autoencoder.model_v2 import AutoencoderV2

logger = get_logger(__name__)

AE_V2_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "autoencoder_v2")
os.makedirs(AE_V2_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(AE_V2_OUTPUT_DIR, "model.pth")

# Training hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_TRAIN_IMAGES = 0  # 0 = use all images (GPU)
SSIM_LOSS_WEIGHT = 0.3  # weight for SSIM loss component


class AugmentedImageDataset(Dataset):
    """
    Training dataset with data augmentation.

    Augmentations:
      - Random horizontal flip
      - Random vertical flip (small probability)
      - Random rotation (±10°)
      - Colour jitter (brightness, contrast, saturation)
      - Random affine (slight translation/scale)
    """

    def __init__(self, image_data: list[dict], img_h: int, img_w: int):
        self.image_data = image_data
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_h, img_w)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]["Image Path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, image


class CombinedLoss(nn.Module):
    """MSE + (1 - SSIM) weighted loss."""

    def __init__(self, ssim_weight: float = 0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_vals = compute_ssim_batch(pred, target)  # (B,)
        ssim_loss = 1.0 - ssim_vals.mean()
        return mse_loss + self.ssim_weight * ssim_loss


def train_autoencoder_v2(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> list[float]:
    """
    Train AutoencoderV2 with combined MSE+SSIM loss, AdamW, cosine LR.
    Returns list of average loss per epoch.
    """
    model.to(device)
    criterion = CombinedLoss(ssim_weight=SSIM_LOSS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=True)
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = criterion(preds, imgs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)
        logger.info(
            "Epoch [%d/%d]  Loss: %.6f  LR: %.2e",
            epoch + 1,
            num_epochs,
            epoch_loss,
            scheduler.get_last_lr()[0],
        )

    return history


def main():
    set_seed()
    ensure_dataset()

    logger.info("Loading dataset...")
    categories = get_categories(DATASET_PATH)
    image_data = collect_image_paths(DATASET_PATH, categories)
    image_data, _ = validate_images(image_data)
    logger.info("Valid training images: %d", len(image_data))

    if MAX_TRAIN_IMAGES > 0 and len(image_data) > MAX_TRAIN_IMAGES:
        random.shuffle(image_data)
        image_data = image_data[:MAX_TRAIN_IMAGES]
        logger.info("Subsampled to %d images", MAX_TRAIN_IMAGES)
    else:
        logger.info("Using all %d training images", len(image_data))

    train_dataset = AugmentedImageDataset(image_data, IMG_HEIGHT, IMG_WIDTH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_dataset) > 0:
        sample, _ = train_dataset[0]
        logger.info(
            "Sample - shape: %s, dtype: %s, min: %.2f, max: %.2f",
            sample.shape,
            sample.dtype,
            sample.min(),
            sample.max(),
        )

    model = AutoencoderV2(bottleneck_channels=128, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    latent_dim = 128 * 8 * 8
    logger.info(
        "AutoencoderV2 - params: %d, latent dim: %d (compression: %.0f:1)",
        total_params,
        latent_dim,
        (3 * 256 * 256) / latent_dim,
    )

    logger.info("Training AutoencoderV2 (%d epochs) on %s...", NUM_EPOCHS, DEVICE)
    loss_history = train_autoencoder_v2(
        model,
        train_loader,
        DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Model saved to: %s", MODEL_SAVE_PATH)
    logger.info("Final loss: %.6f", loss_history[-1])

    return model, loss_history


if __name__ == "__main__":
    main()

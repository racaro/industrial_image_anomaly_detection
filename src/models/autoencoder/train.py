"""Training pipeline for the Convolutional Autoencoder."""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    ensure_dataset,
    set_seed,
)
from src.dataset import prepare_training_data
from src.logger import get_logger
from src.models.autoencoder import Autoencoder

logger = get_logger(__name__)

AE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "autoencoder")
os.makedirs(AE_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(AE_OUTPUT_DIR, "model.pth")


def train_autoencoder(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train the autoencoder with MSELoss + Adam.
    Returns a list of average loss per epoch.
    """
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)
        logger.info("Epoch [%d/%d]  Average loss: %.6f", epoch + 1, num_epochs, epoch_loss)

    return history


def main():
    set_seed()
    ensure_dataset()

    train_loader = prepare_training_data(DATASET_PATH, BATCH_SIZE)

    logger.info("Training autoencoder (%d epochs) on %s...", NUM_EPOCHS, DEVICE)
    model = Autoencoder()
    loss_history = train_autoencoder(
        model,
        train_loader,
        DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Model saved to: %s", MODEL_SAVE_PATH)

    return model, loss_history


if __name__ == "__main__":
    model, loss_history = main()

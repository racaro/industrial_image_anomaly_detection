"""
Training pipeline for the Convolutional Autoencoder.

Usage:
    python -m src.models.autoencoder.train
    python src/models/autoencoder/train.py
"""

import os
import sys

# Ensure project root is on sys.path so absolute imports work
# regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE, DATASET_PATH, DEVICE, IMG_HEIGHT, IMG_WIDTH,
    LEARNING_RATE, NUM_EPOCHS, NUM_IMAGES_TO_MOVE, OUTPUTS_DIR,
    ensure_dataset,
)
from src.dataset import (
    AnomalyImageDataset, balance_test_good, build_distribution_df,
    collect_image_paths, count_images, get_categories,
    print_distribution_summary, validate_images,
)
from src.models.autoencoder import Autoencoder

# ──────────────────────────────────────────────
# OUTPUT PATHS
# ──────────────────────────────────────────────

AE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "autoencoder")
os.makedirs(AE_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(AE_OUTPUT_DIR, "model.pth")


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

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

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{num_epochs}",
                    unit="batch", leave=True)
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
        print(f"  Epoch [{epoch + 1}/{num_epochs}]  Average loss: {epoch_loss:.6f}")

    return history


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    # --- Ensure dataset is extracted ---
    ensure_dataset()

    # --- Dataset exploration ---
    print("Loading dataset categories...")
    categories = get_categories(DATASET_PATH)
    print(f"  Categories found: {len(categories)}")

    image_counts = count_images(DATASET_PATH, categories)
    df_distribution = build_distribution_df(image_counts)
    print_distribution_summary(df_distribution, title="Initial image distribution")

    # --- Balancing: move images to test/good if missing ---
    cats_no_test_good = df_distribution.loc[
        df_distribution["Test Good"] == 0, "Category"
    ].tolist()

    if cats_no_test_good:
        print(f"Categories without test/good: {cats_no_test_good}")
        balance_test_good(DATASET_PATH, cats_no_test_good, NUM_IMAGES_TO_MOVE)

        image_counts = count_images(DATASET_PATH, categories)
        df_distribution = build_distribution_df(image_counts)
        print_distribution_summary(df_distribution, title="Distribution after balancing")

    # --- Collect paths ---
    image_data = collect_image_paths(DATASET_PATH, categories)
    df_images = pd.DataFrame(image_data)
    print(f"Total train/good images collected: {len(df_images)}")
    print(df_images.head().to_markdown(index=False))

    # --- Validate images ---
    image_data, df_validation = validate_images(image_data)

    # --- PyTorch Dataset and DataLoader ---
    train_dataset = AnomalyImageDataset(image_data, IMG_HEIGHT, IMG_WIDTH)
    num_workers = 0 if os.name == "nt" else 4
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    if len(train_dataset) > 0:
        sample, _ = train_dataset[0]
        print(f"\nSample – shape: {sample.shape}, dtype: {sample.dtype}, "
              f"min: {sample.min():.2f}, max: {sample.max():.2f}")

    # --- Train ---
    print(f"\nTraining autoencoder ({NUM_EPOCHS} epochs) on {DEVICE}...\n")
    model = Autoencoder()
    loss_history = train_autoencoder(
        model, train_loader, DEVICE,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
    )

    # --- Save model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    return model, loss_history


if __name__ == "__main__":
    model, loss_history = main()

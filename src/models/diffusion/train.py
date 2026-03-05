"""
Training script for DiffusionModel (DDPM) anomaly detection.

Implements proper DDPM training:
    1. Sample clean image x_0 from train/good.
    2. Sample random timestep t ~ Uniform(0, T).
    3. Sample noise ε ~ N(0, I).
    4. Create noisy image x_t via forward diffusion.
    5. Predict noise ε̂ = UNet(x_t, t).
    6. Minimise MSE(ε, ε̂).

Saves best model (lowest validation loss) to outputs/diffusion/model.pth.
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
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
    AnomalyImageDataset,
    balance_test_good,
    build_distribution_df,
    collect_image_paths,
    count_images,
    get_categories,
    print_distribution_summary,
    validate_images,
)
from src.logger import get_logger
from src.models.diffusion.model import DiffusionModel

logger = get_logger(__name__)

DIFFUSION_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "diffusion")
MODEL_SAVE_PATH = os.path.join(DIFFUSION_OUTPUT_DIR, "model.pth")
DIFFUSION_LR = 2e-4
DIFFUSION_EPOCHS = 40  # GPU-optimized training


def train_diffusion(
    model: DiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = DIFFUSION_EPOCHS,
    lr: float = DIFFUSION_LR,
) -> list[float]:
    """
    Train the diffusion model with ε-prediction MSE loss.

    Args:
        model: DiffusionModel instance.
        dataloader: Training DataLoader (normal images only).
        device: Torch device.
        num_epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        List of average loss per epoch.
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history: list[float] = []
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        progress = tqdm(dataloader, desc=f"  Epoch {epoch:03d}/{num_epochs}", unit="batch")
        for batch_imgs, _ in progress:
            batch_imgs = batch_imgs.to(device)
            batch_size = batch_imgs.shape[0]

            # Sample random timesteps
            t = torch.randint(0, model.timesteps, (batch_size,), device=device)

            # Forward diffusion: add noise
            x_t, noise = model.q_sample(batch_imgs, t)

            # Predict noise
            noise_pred = model(x_t, t)

            # Loss
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            num_samples += batch_size
            progress.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = epoch_loss / num_samples
        history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %03d/%03d | Loss: %.6f | LR: %.2e",
            epoch,
            num_epochs,
            avg_loss,
            current_lr,
        )

        # Save sample reconstructions periodically
        if epoch % 10 == 0 or epoch == num_epochs:
            _save_sample_reconstructions(model, dataloader, device, epoch)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info("  ✓ Saved best model (loss=%.6f)", best_loss)

    logger.info("Training complete. Best loss: %.6f", best_loss)
    return history


@torch.no_grad()
def _save_sample_reconstructions(
    model: DiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """Save a grid of original / noisy / reconstructed images."""
    model.eval()
    sample = next(iter(dataloader))[0][:8].to(device)

    # Add noise at the inference noise level
    t_start = int(model.timesteps * model.noise_level)
    t_tensor = torch.full((sample.shape[0],), t_start - 1, device=device, dtype=torch.long)
    noisy, _ = model.q_sample(sample, t_tensor)

    # Reconstruct
    recon = model.reconstruct(sample)

    # Build grid: row 1 = original, row 2 = noisy, row 3 = reconstructed
    grid = torch.cat([sample, noisy.clamp(0, 1), recon], dim=0)
    grid_path = os.path.join(DIFFUSION_OUTPUT_DIR, f"recon_epoch{epoch:03d}.png")
    save_image(grid, grid_path, nrow=8, normalize=False)
    logger.info("  Saved reconstructions → %s", grid_path)
    model.train()


def main() -> None:
    """Entry point: prepare data and train the diffusion model."""
    set_seed()
    os.makedirs(DIFFUSION_OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  DIFFUSION MODEL TRAINING")
    logger.info("=" * 60)

    ensure_dataset()
    categories = get_categories(DATASET_PATH)
    image_counts = count_images(DATASET_PATH, categories)
    df_dist = build_distribution_df(image_counts)
    print_distribution_summary(df_dist, "Dataset Distribution (Before Balancing)")

    # Balance categories missing test/good
    cats_missing = [cat for cat, c in image_counts.items() if c["test_good"] == 0]
    if cats_missing:
        logger.info("Balancing %d categories with no test/good images...", len(cats_missing))
        balance_test_good(DATASET_PATH, cats_missing, num_to_move=100)
        image_counts = count_images(DATASET_PATH, categories)
        df_dist = build_distribution_df(image_counts)
        print_distribution_summary(df_dist, "Dataset Distribution (After Balancing)")

    # Validate images
    image_data = collect_image_paths(DATASET_PATH, categories, subfolder=os.path.join("train", "good"))
    image_data, _df_validation = validate_images(image_data)
    logger.info("Training images (after validation): %d", len(image_data))

    # Subsample for CPU-feasible training (use all data on GPU)
    max_train_images = 4000
    if DEVICE.type == "cpu" and len(image_data) > max_train_images:
        import random as _rng

        _rng.seed(42)
        image_data = _rng.sample(image_data, max_train_images)
        logger.info("Subsampled to %d images for CPU training", max_train_images)

    dataset = AnomalyImageDataset(image_data, IMG_HEIGHT, IMG_WIDTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = DiffusionModel(
        timesteps=1000,
        schedule="cosine",
        inference_steps=20,
        noise_level=0.25,
    )
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logger.info("Device: %s", DEVICE)

    history = train_diffusion(model, dataloader, DEVICE, num_epochs=DIFFUSION_EPOCHS, lr=DIFFUSION_LR)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(history) + 1), history, "b-", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Diffusion Model Training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIFFUSION_OUTPUT_DIR, "training_loss.png"), dpi=150)
    plt.close()
    logger.info("Training loss plot saved.")
    logger.info("Done. Model saved to %s", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()

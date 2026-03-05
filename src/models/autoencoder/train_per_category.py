"""
Per-category Autoencoder training pipeline.

Trains one independent AE V1 model per dataset category, allowing each model
to specialize in the normal appearance of a single product type.
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    NUM_WORKERS,
    OUTPUTS_DIR,
    ensure_dataset,
    set_seed,
)
from src.dataset import (
    AnomalyImageDataset,
    collect_image_paths,
    get_categories,
    validate_images,
)
from src.logger import get_logger
from src.models.autoencoder import Autoencoder

logger = get_logger(__name__)

PER_CATEGORY_EPOCHS = 30
"""Default epochs per category. Less data per category → faster convergence."""

PER_CATEGORY_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "autoencoder_per_category")
"""Root directory for all per-category models."""


def train_single_category(
    category: str,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    save_dir: str,
) -> dict:
    """
    Train one Autoencoder on a single category's training data.

    Args:
        category: Category name (e.g. 'bottle').
        dataloader: DataLoader with that category's train/good images.
        device: Torch device (cuda / cpu).
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        save_dir: Directory to save model.pth and training_log.json.

    Returns:
        Dict with training summary (category, epochs, loss_history, etc.).
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pth")

    model = Autoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"  [{category}] Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
        )
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
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "[%s] Epoch [%d/%d]  loss: %.6f",
                category,
                epoch + 1,
                num_epochs,
                epoch_loss,
            )

    elapsed = time.time() - start_time

    # Save model
    torch.save(model.state_dict(), model_path)
    logger.info("[%s] Model saved → %s", category, model_path)

    # Training summary
    summary = {
        "category": category,
        "num_images": len(dataloader.dataset),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": dataloader.batch_size,
        "final_loss": round(loss_history[-1], 6),
        "initial_loss": round(loss_history[0], 6),
        "elapsed_seconds": round(elapsed, 1),
        "loss_history": [round(val, 6) for val in loss_history],
    }

    # Save training log
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main(
    selected_categories: list[str] | None = None,
    num_epochs: int = PER_CATEGORY_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Train one Autoencoder per category.

    Args:
        selected_categories: Specific categories to train. None = all.
        num_epochs: Epochs per category.
        learning_rate: Adam learning rate.
        batch_size: Batch size for DataLoader.

    Returns:
        List of training summaries (one per category).
    """
    set_seed()
    ensure_dataset()

    all_categories = get_categories(DATASET_PATH)
    logger.info("Dataset categories found: %d", len(all_categories))

    # Filter by user selection
    if selected_categories:
        invalid = [c for c in selected_categories if c not in all_categories]
        if invalid:
            raise ValueError(f"Unknown categories: {invalid}. Available: {all_categories}")
        categories = selected_categories
    else:
        categories = all_categories

    logger.info(
        "Will train %d per-category models (%d epochs each) on %s",
        len(categories),
        num_epochs,
        DEVICE,
    )

    use_pin_memory = torch.cuda.is_available()
    summaries: list[dict] = []
    total_start = time.time()

    for i, category in enumerate(categories, 1):
        logger.info("=" * 60)
        logger.info(
            "  [%d/%d] Training: %s",
            i,
            len(categories),
            category.upper(),
        )
        logger.info("=" * 60)

        # Collect images for this category only
        image_data = collect_image_paths(DATASET_PATH, [category], subfolder=os.path.join("train", "good"))

        if not image_data:
            logger.warning("[%s] No training images found — skipping.", category)
            continue

        # Validate images
        image_data, _ = validate_images(image_data)
        logger.info("[%s] Valid training images: %d", category, len(image_data))

        if len(image_data) < 5:
            logger.warning("[%s] Too few images (%d) — skipping.", category, len(image_data))
            continue

        # DataLoader
        dataset = AnomalyImageDataset(image_data, IMG_HEIGHT, IMG_WIDTH)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=use_pin_memory,
        )

        # Train
        save_dir = os.path.join(PER_CATEGORY_OUTPUT_DIR, category)
        summary = train_single_category(
            category=category,
            dataloader=dataloader,
            device=DEVICE,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_dir=save_dir,
        )
        summaries.append(summary)

        logger.info(
            "[%s] Done — loss: %.6f → %.6f  (%d images, %.0fs)",
            category,
            summary["initial_loss"],
            summary["final_loss"],
            summary["num_images"],
            summary["elapsed_seconds"],
        )

    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info("  PER-CATEGORY TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Models trained:  %d / %d", len(summaries), len(categories))
    logger.info("Total time:      %.1f min", total_elapsed / 60)
    logger.info("Output dir:      %s", PER_CATEGORY_OUTPUT_DIR)

    for s in summaries:
        logger.info(
            "  %-15s  images: %4d  loss: %.6f → %.6f  (%.0fs)",
            s["category"],
            s["num_images"],
            s["initial_loss"],
            s["final_loss"],
            s["elapsed_seconds"],
        )

    # Save global summary
    global_summary = {
        "total_categories": len(summaries),
        "total_time_seconds": round(total_elapsed, 1),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(DEVICE),
        "categories": summaries,
    }
    summary_path = os.path.join(PER_CATEGORY_OUTPUT_DIR, "training_summary.json")
    os.makedirs(PER_CATEGORY_OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    logger.info("Global summary saved → %s", summary_path)

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one Autoencoder (AE V1) per dataset category.")
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=None,
        help="Specific categories to train. Default: all.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=PER_CATEGORY_EPOCHS,
        help=f"Epochs per category (default: {PER_CATEGORY_EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    main(
        selected_categories=args.categories,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

"""
Training pipeline for GAN-based anomaly detection.

Architecture:
    - Generator (G): Encoder-decoder identical to the Autoencoder
    - Discriminator (D): PatchGAN with spectral normalization

Training:
    G loss = λ_adv * BCE(D(G(x)), 1) + λ_rec * MSE(x, G(x))
    D loss = BCE(D(x), 1) + BCE(D(G(x)).detach()), 0)

Anomaly score at test time: MSE(x, G(x))
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    ensure_dataset,
    set_seed,
)
from src.dataset import prepare_training_data
from src.logger import get_logger
from src.models.gan import Discriminator, Generator

logger = get_logger(__name__)

GAN_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "gan")
os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
GENERATOR_SAVE_PATH = os.path.join(GAN_OUTPUT_DIR, "generator.pth")
DISCRIMINATOR_SAVE_PATH = os.path.join(GAN_OUTPUT_DIR, "discriminator.pth")

LAMBDA_ADV = 1.0  # adversarial loss weight
LAMBDA_REC = 50.0  # reconstruction loss weight (high → prioritize reconstruction)
LR_G = 1e-4  # generator learning rate
LR_D = 1e-4  # discriminator learning rate


def weights_init(m):
    """Apply custom weight initialization (DCGAN convention)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    lambda_adv: float = 1.0,
    lambda_rec: float = 50.0,
) -> dict[str, list[float]]:
    """
    Train GAN for anomaly detection.

    Returns dict with loss histories:
        - 'g_loss': total generator loss per epoch
        - 'd_loss': discriminator loss per epoch
        - 'g_rec': reconstruction (MSE) loss per epoch
        - 'g_adv': adversarial loss per epoch
    """
    generator.to(device)
    discriminator.to(device)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    history = {"g_loss": [], "d_loss": [], "g_rec": [], "g_adv": []}

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        run_g_loss = 0.0
        run_d_loss = 0.0
        run_g_rec = 0.0
        run_g_adv = 0.0
        n_samples = 0

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=True)

        for real_imgs, _ in pbar:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            label_real = torch.ones(bs, 1, device=device)
            label_fake = torch.zeros(bs, 1, device=device)

            # ── Train Discriminator ──
            fake_imgs = generator(real_imgs).detach()

            d_real = discriminator(real_imgs)
            d_fake = discriminator(fake_imgs)

            loss_d_real = criterion_bce(d_real, label_real)
            loss_d_fake = criterion_bce(d_fake, label_fake)
            loss_d = (loss_d_real + loss_d_fake) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ── Train Generator ──
            fake_imgs = generator(real_imgs)
            d_fake_for_g = discriminator(fake_imgs)

            loss_adv = criterion_bce(d_fake_for_g, label_real)
            loss_rec = criterion_mse(fake_imgs, real_imgs)
            loss_g = lambda_adv * loss_adv + lambda_rec * loss_rec

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # ── Accumulate ──
            run_g_loss += loss_g.item() * bs
            run_d_loss += loss_d.item() * bs
            run_g_rec += loss_rec.item() * bs
            run_g_adv += loss_adv.item() * bs
            n_samples += bs

            pbar.set_postfix(
                D=f"{loss_d.item():.4f}",
                G=f"{loss_g.item():.4f}",
                rec=f"{loss_rec.item():.6f}",
            )

        # Epoch averages
        n = len(dataloader.dataset)
        ep_g = run_g_loss / n
        ep_d = run_d_loss / n
        ep_rec = run_g_rec / n
        ep_adv = run_g_adv / n

        history["g_loss"].append(ep_g)
        history["d_loss"].append(ep_d)
        history["g_rec"].append(ep_rec)
        history["g_adv"].append(ep_adv)

        logger.info(
            "Epoch [%d/%d]  D_loss: %.6f  G_loss: %.6f  Rec: %.6f  Adv: %.6f",
            epoch + 1,
            num_epochs,
            ep_d,
            ep_g,
            ep_rec,
            ep_adv,
        )

    return history


def main():
    set_seed()
    ensure_dataset()

    train_loader = prepare_training_data(DATASET_PATH, BATCH_SIZE)

    generator = Generator()
    discriminator = Discriminator()
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    logger.info("Training GAN (%d epochs) on %s...", NUM_EPOCHS, DEVICE)
    logger.info("  λ_adv=%.1f, λ_rec=%.1f, lr_G=%.1e, lr_D=%.1e", LAMBDA_ADV, LAMBDA_REC, LR_G, LR_D)

    history = train_gan(
        generator,
        discriminator,
        train_loader,
        DEVICE,
        num_epochs=NUM_EPOCHS,
        lr_g=LR_G,
        lr_d=LR_D,
        lambda_adv=LAMBDA_ADV,
        lambda_rec=LAMBDA_REC,
    )

    torch.save(generator.state_dict(), GENERATOR_SAVE_PATH)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_SAVE_PATH)
    logger.info("Generator saved to:     %s", GENERATOR_SAVE_PATH)
    logger.info("Discriminator saved to: %s", DISCRIMINATOR_SAVE_PATH)

    return generator, discriminator, history


if __name__ == "__main__":
    generator, discriminator, history = main()

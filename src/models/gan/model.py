"""
GAN-based Anomaly Detection – Generator + Discriminator.

The Generator has the same encoder-decoder architecture as the Autoencoder
so that reconstruction-based anomaly scores are directly comparable.

The Discriminator is a PatchGAN-style CNN that classifies whether
image patches are real or reconstructed.

Training approach:
    - G minimises:  λ_adv * L_adv  +  λ_rec * L_rec
    - D minimises:  standard binary cross-entropy (real=1, fake=0)
    - Anomaly score at test time = MSE(x, G(x))  (same as AE)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator (encoder-decoder) identical to the Autoencoder.
    Input: (B, 3, 256, 256)  →  Output: (B, 3, 256, 256)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    Input: (B, 3, 256, 256)  →  Output: (B, 1) probability (real or fake).

    Architecture:
        4 strided conv layers progressively reduce spatial dims,
        followed by adaptive average pooling and a single sigmoid output.
    Uses LeakyReLU (slope=0.2) and spectral normalization for stability.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

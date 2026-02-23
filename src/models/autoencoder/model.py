"""
Convolutional Autoencoder for anomaly detection.

Uses padding=1 on all layers to avoid border artifacts
and BatchNorm to stabilize training.
Encoder:  256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16  (latent 256x16x16)
Decoder:  16x16  -> 32x32  -> 64x64 -> 128x128 -> 256x256
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Convolutional autoencoder for anomaly detection.

    Architecture:
        - 4-layer encoder with strided Conv2d + BatchNorm + ReLU
        - 4-layer decoder with ConvTranspose2d + BatchNorm + ReLU
        - Sigmoid output to keep values in [0, 1]
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

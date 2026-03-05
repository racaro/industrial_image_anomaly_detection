"""
Improved Convolutional Autoencoder (V2) for Anomaly Detection.

Key improvements over V1:
  - **Tighter bottleneck**: 5 downsample layers → 8×8 spatial, 128 channels
    Latent dim: 128×8×8 = 8,192  (vs V1: 256×16×16 = 65,536)
    Compression ratio: ~24:1  (vs V1: ~3:1)
  - **Dropout** in the bottleneck to prevent memorisation
  - **LeakyReLU** instead of ReLU for better gradient flow
  - Configurable bottleneck size

Architecture:
  Encoder:  256→128→64→32→16→8   channels: 3→32→64→128→256→bottleneck
  Decoder:  8→16→32→64→128→256   channels: bottleneck→256→128→64→32→3
"""

import torch.nn as nn


class AutoencoderV2(nn.Module):
    """
    Deep convolutional autoencoder with tight bottleneck.

    Args:
        bottleneck_channels: Number of channels at the bottleneck (default: 128).
            Latent dim = bottleneck_channels × 8 × 8.
        dropout: Dropout probability applied at the bottleneck (default: 0.1).
    """

    def __init__(self, bottleneck_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels

        self.encoder = nn.Sequential(
            # 256×256 → 128×128
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 128×128 → 64×64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×64 → 32×32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8  (bottleneck)
            nn.Conv2d(256, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            # 8×8 → 16×16
            nn.ConvTranspose2d(bottleneck_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×64 → 128×128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 128×128 → 256×256
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(self.encoder(x))

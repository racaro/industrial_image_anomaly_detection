"""
src.models - Neural network architectures for anomaly detection.

Subpackages:
    autoencoder     Convolutional Autoencoder
    gan             Generator + PatchGAN Discriminator
"""

from src.models.autoencoder import Autoencoder
from src.models.gan import Discriminator, Generator

__all__ = ["Autoencoder", "Discriminator", "Generator"]

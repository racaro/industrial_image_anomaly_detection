"""
src.models - Neural network architectures for anomaly detection.

Subpackages:
    autoencoder     Convolutional Autoencoder (V1 and V2)
    gan             Generator + PatchGAN Discriminator
    diffusion       Denoising Diffusion Probabilistic Model
    patchcore       PatchCore with pre-trained features
"""

from src.models.autoencoder import Autoencoder, AutoencoderV2
from src.models.diffusion import DiffusionModel
from src.models.gan import Discriminator, Generator

__all__ = ["Autoencoder", "AutoencoderV2", "DiffusionModel", "Discriminator", "Generator"]

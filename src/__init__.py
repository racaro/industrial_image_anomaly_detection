"""
src – Source code for anomaly detection on industrial images.

Modules:
    config              Paths, hyperparameters, device selection
    dataset             Dataset exploration, validation, PyTorch datasets
    metrics             SSIM computation
    evaluate            Unified evaluation (--model autoencoder|gan)

Model sub-packages (src.models.*):
    autoencoder         Convolutional Autoencoder (model + training)
    gan                 Generator + PatchGAN Discriminator (model + training)
"""

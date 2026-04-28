"""
Image preprocessing helpers for CIFAR-10-style models (``32×32``, channel-wise normalization).
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

# Matches ``fetch_cifar10_loader`` defaults: outputs approximately ``[-1, 1]`` per channel.
CIFAR10_NORMALIZE = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def pil_to_cifar_tensor(pil_rgb: Image.Image, target_size: int = 32) -> torch.Tensor:
    """
    Convert a Pillow RGB image into a tensor ``[3,H,W]`` sized for CIFAR-style ResNet demos.
    """
    tfms = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            CIFAR10_NORMALIZE,
        ]
    )

    return tfms(pil_rgb)


__all__ = ["CIFAR10_NORMALIZE", "pil_to_cifar_tensor"]

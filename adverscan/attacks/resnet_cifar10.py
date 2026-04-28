"""
ResNet-18 for ``32×32`` CIFAR-10 (no max-pool stem; ``3×3`` stride-1 convolution).

Loads published ``.th/.pth`` checkpoints keyed by plain ``conv1``, ``layerN.*``, ``fc`` names.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def _sanitize_state_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {str(k).replace("module.", ""): v for k, v in raw.items()}


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    """CIFAR ResNet macro-structure with global average pooling and linear logits."""

    def __init__(self, block: type[BasicBlock], layers: tuple[int, int, int, int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers_ls: list[nn.Module] = []
        for s in strides:

            layers_ls.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers_ls)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatially pooled representation before logits (dimension ``512``).

        Used by baseline detectors relying on Euclidean neighborhoods (e.g. LID proxies).
        """
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc(self.embed(x))


def resnet18_cifar10(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-18 for CIFAR-size inputs."""
    return ResNetCIFAR(BasicBlock, (2, 2, 2, 2), num_classes=num_classes)


def resolve_state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a ``state_dict`` from hub-style payloads or raw tensor maps."""
    if isinstance(payload, nn.Module):

        return _sanitize_state_dict(dict(payload.state_dict().items()))

    if not isinstance(payload, dict):

        raise RuntimeError("Checkpoint payload must map or nn.Module.")

    for key in ("state_dict", "model", "net", "weights"):

        inner = payload.get(key)
        if isinstance(inner, dict) and inner:

            first = next(iter(inner.values()))

            if torch.is_tensor(first):

                return _sanitize_state_dict(dict(inner.items()))

    first_top = next(iter(payload.values()), None)

    if torch.is_tensor(first_top):

        return _sanitize_state_dict({k: v for k, v in payload.items() if torch.is_tensor(v)})

    raise RuntimeError("Unable to infer state_dict from checkpoint blob.")


def load_cifar10_resnet18_weights(model: nn.Module, checkpoint_path: str) -> bool:
    """
    Load weights from ``.pth``/``.th``; return ``True`` if parameters align (strict or partial success).
    """
    if not os.path.isfile(checkpoint_path):

        return False

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    try:

        state = resolve_state_dict_from_checkpoint(payload)

    except RuntimeError:

        return False

    missing, unexpected = model.load_state_dict(state, strict=False)

    return len(missing) == 0 and len(unexpected) == 0


def build_pretrained_cifar10_resnet18(weights_path: str | None = None) -> tuple[nn.Module, bool]:
    """
    Build ResNet-18 and optionally load pretrained CIFAR-10 checkpoints.

    Tries paths in order: explicit ``weights_path``, then ``ADVERSCAN_VICTIM_CHECKPOINT``,
    then ``ADVERSCAN_CIFAR10_RESNET18``. The first readable file whose state dict aligns
    with the architecture activates weights.
    """
    net = resnet18_cifar10(num_classes=10)

    candidate_paths: list[str] = []
    if weights_path:
        candidate_paths.append(weights_path)

    extra_env_candidates = ("ADVERSCAN_VICTIM_CHECKPOINT", "ADVERSCAN_CIFAR10_RESNET18")

    for candidate_key in extra_env_candidates:
        cand = os.getenv(candidate_key)

        if cand and cand not in candidate_paths:

            candidate_paths.append(cand)

    for path_candidate in candidate_paths:

        if os.path.isfile(path_candidate) and load_cifar10_resnet18_weights(net, path_candidate):

            return net, True

    return net, False


__all__ = [
    "BasicBlock",
    "ResNetCIFAR",
    "build_pretrained_cifar10_resnet18",
    "load_cifar10_resnet18_weights",
    "resnet18_cifar10",
    "resolve_state_dict_from_checkpoint",
]

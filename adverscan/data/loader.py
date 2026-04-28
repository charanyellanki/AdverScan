"""
Dataset loaders bridging image (CIFAR-10) and tabular benchmarking suites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def trivial_cifar_cnn(num_classes: int = 10) -> nn.Module:
    """Small deterministic CNN skeleton used for prototyping attacks/detectors."""

    class TinyCifarCNN(nn.Module):
        """ConvNet intended for experimentation on ``32×32`` RGB tensors."""

        def __init__(self, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(32 * 8 * 8, classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = torch.flatten(self.features(x), start_dim=1)
            return self.classifier(feats)

    return TinyCifarCNN(classes=num_classes)


def fetch_cifar10_loader(
    root: str = "./datasets",
    *,
    batch_size: int = 128,
    train: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    normalize_mean: Iterable[float] = (0.5, 0.5, 0.5),
    normalize_std: Iterable[float] = (0.5, 0.5, 0.5),
) -> DataLoader:
    """
    Return a torchvision DataLoader emitting ``[-1,1]`` tensors by default normalization.

    Parameters
    ----------
    root
        Download/cache folder for torchvision ``CIFAR10``.
    batch_size
        Loader batch cardinality.
    train
        Whether to load training split versus evaluation split.
    num_workers
        PyTorch multiprocessing loader workers count.
    pin_memory
        Whether to allocate CUDA pinned tensors when CUDA available.
    normalize_mean/std
        Channel-wise normalization parameters passed to ``transforms.Compose``.

    Returns
    -------
    DataLoader
        Iterable yielding tuples ``(images, labels)`` with ``torch.float`` tensors ``[N,3,32,32]``.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(tuple(normalize_mean), tuple(normalize_std)),
        ]
    )

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)


@dataclass(slots=True)
class TabularBenchConfig:
    """Configuration container for deterministic tabular dataset splits."""

    test_size: float = 0.2
    random_state: int = 1337


def benchmark_tabular_splits(config: TabularBenchConfig | None = None) -> dict[str, Any]:
    """
    Prepare Wisconsin Breast Cancer splits as lightweight tabular benchmark metadata.

    The resulting dictionary contains numpy tensors ``features_{train,val,test}`` ready for sklearn tooling.

    Parameters
    ----------
    config
        Optional split knobs; defaults to reproducible stratified partitioning.

    Returns
    -------
    dict
        Structured payload with ``datasets.SKLEARN_FETCH`` metadata alongside numpy arrays ``X_*`` ``y_*``.
    """
    cfg = config or TabularBenchConfig()
    data_bundle = load_breast_cancer()
    features = data_bundle.data.astype(np.float32)
    labels = data_bundle.target.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)

    splitter = train_test_split(
        X_train_val_scaled,
        y_train_val,
        test_size=0.25,
        random_state=cfg.random_state,
        stratify=y_train_val,
    )

    X_train, X_val, y_train, y_val = splitter

    return {
        "name": "sklearn.datasets.load_breast_cancer",
        "feature_names": data_bundle.feature_names,
        "target_names": list(data_bundle.target_names),
        "feature_scaler_mean": scaler.mean_,
        "feature_scaler_scale": scaler.scale_,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def ndarray_to_loader(
    features: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Compose a ``Dataset`` wrapping numpy arrays backed by ``TensorDataset``."""
    subset = TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels.astype(np.int64)))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


BenchSplitKey = Literal["train", "val", "test"]


__all__ = [
    "BenchSplitKey",
    "TabularBenchConfig",
    "benchmark_tabular_splits",
    "fetch_cifar10_loader",
    "ndarray_to_loader",
    "trivial_cifar_cnn",
]

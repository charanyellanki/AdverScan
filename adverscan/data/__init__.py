"""Dataset helpers for benchmarking adversarial-detection tooling."""

from adverscan.data.loader import (
    BenchSplitKey,
    TabularBenchConfig,
    benchmark_tabular_splits,
    fetch_cifar10_loader,
    ndarray_to_loader,
    trivial_cifar_cnn,
)

__all__ = [
    "BenchSplitKey",
    "TabularBenchConfig",
    "benchmark_tabular_splits",
    "fetch_cifar10_loader",
    "ndarray_to_loader",
    "trivial_cifar_cnn",
]

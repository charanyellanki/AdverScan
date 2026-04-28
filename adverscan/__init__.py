"""
AdverScan: adversarial-example detection tooling with attack generation,
feature-based detectors, and evaluation harnesses.

This package organizes attack implementations, detectors, datasets, and API
utilities under a single import path (``import adverscan``).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("adverscan")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

"""
Unified adversarial attack interface with :class:`AttackRunner` orchestration.

Victim models are assumed to be differentiable ResNet-18 style CIFAR-10 classifiers
(see :mod:`adverscan.attacks.resnet_cifar10`) but any ``nn.Module`` mapping to logits suffices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

import torch
import torch.nn as nn

from . import cw as cw_module
from . import fgsm as fgsm_module
from . import pgd as pgd_module

from .resnet_cifar10 import ResNetCIFAR, build_pretrained_cifar10_resnet18, resnet18_cifar10
from .result import AttackResult

AttackName = Literal["fgsm", "pgd", "cw"]


class AttackCallable(Protocol):
    """Signature shared by FGSM, PGD, and CW modules."""

    def __call__(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        true_label: torch.Tensor,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> AttackResult:
        ...


_REGISTERED: dict[AttackName, AttackCallable] = {
    "fgsm": fgsm_module.fgsm,
    "pgd": pgd_module.pgd_attack,
    "cw": cw_module.cw_attack,
}


def register_attack(name: AttackName, fn: AttackCallable) -> None:
    """Register or replace an attack callable under ``name``."""
    _REGISTERED[name] = fn


def get_attack(name: AttackName) -> AttackCallable:
    """Resolve callable returning :class:`AttackResult`."""
    if name not in _REGISTERED:
        raise KeyError(f"Unknown attack: {name}. Available: {list(_REGISTERED)}")
    return _REGISTERED[name]


def available_attacks() -> tuple[AttackName, ...]:
    """Return registered attack identifiers."""
    return tuple(_REGISTERED.keys())


class FGSMKw(TypedDict, total=False):
    """Forwarded keyword arguments for FGSM."""

    clamp: tuple[float, float]
    criterion: Any
    targeted: bool


class PGDKw(TypedDict, total=False):
    """Forwarded keyword arguments for PGD."""

    steps: int
    alpha: float
    random_start: float
    clamp: tuple[float, float]
    targeted: bool
    criterion: Any


class CWKw(TypedDict, total=False):
    """Forwarded keyword arguments for Carlini–Wagner objective."""

    steps: int
    learning_rate: float
    clamp: tuple[float, float]
    targeted: bool
    c: float
    kappa: float


@dataclass(slots=True)
class AttackRunner:
    """
    Thin facade mapping attack names to calibrated callables.

    ``default_clamp`` is forwarded when callers omit explicit ``clamp``.
    """

    default_clamp: tuple[float, float] | None = field(default=None)

    def run(
        self,
        name: AttackName,
        model: nn.Module,
        input_tensor: torch.Tensor,
        true_label: torch.Tensor,
        *,
        epsilon: float,
        clamp: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> AttackResult:
        """
        Execute attack ``name`` returning structured :class:`AttackResult`.

        Parameters
        ----------
        model
            Victim differentiable through ``input_tensor`` (e.g. ResNet-18 CIFAR weights).
        input_tensor
            ``(N, C, H, W)`` clean batch aligned with ``epsilon`` pixel units.
        true_label
            ``(N,)`` ``torch.long`` label vector (ground-truth labels for untargeted attacks).
        epsilon
            Intensity knob: FGSM/PGD L∞ ``ε`` radius; CW initializer scale coupling.
        """
        fn = get_attack(name)
        effective_clamp = clamp if clamp is not None else self.default_clamp

        merged: dict[str, Any] = dict(kwargs)

        if effective_clamp is not None and "clamp" not in merged:

            merged["clamp"] = effective_clamp

        return fn(model, input_tensor, true_label, epsilon=float(epsilon), **merged)


def run_attack(
    name: AttackName,
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: torch.Tensor,
    *,
    epsilon: float,
    clamp: tuple[float, float] | None = None,
    **kwargs: Any,
) -> AttackResult:
    """
    Stateless wrapper around ``get_attack(name)(...)`` returning :class:`AttackResult`.

    For repeated sweeps with a shared clamp policy, prefer :class:`AttackRunner`.
    """
    fn = get_attack(name)

    return fn(model, input_tensor, true_label, epsilon=float(epsilon), clamp=clamp, **kwargs)


__all__ = [
    "AttackCallable",
    "AttackName",
    "AttackResult",
    "AttackRunner",
    "ResNetCIFAR",
    "available_attacks",
    "build_pretrained_cifar10_resnet18",
    "get_attack",
    "register_attack",
    "resnet18_cifar10",
    "run_attack",
]

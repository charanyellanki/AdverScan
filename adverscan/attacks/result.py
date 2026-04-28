"""Structured return type for constrained attack outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttackResult:
    """
    Dual outputs expected by callers who need perturbation bookkeeping.

    ``perturbation_magnitude_l2`` stores per-batch L2 norms :math:`\\|x' - x\\|_2`.
    """

    adversarial_examples: torch.Tensor
    perturbation_magnitude_l2: torch.Tensor

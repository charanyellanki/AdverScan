"""
Projected Gradient Descent (PGD) under L∞ for CIFAR-scale inputs.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from adverscan.attacks.result import AttackResult


def _batch_l2_perturbation(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    diff = (adversarial - clean).view(clean.shape[0], -1)
    return torch.linalg.vector_norm(diff, ord=2, dim=1)


def pgd_attack(
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: torch.Tensor,
    *,
    epsilon: float,
    criterion: Callable[..., torch.Tensor] | None = None,
    steps: int = 10,
    alpha: float | None = None,
    clamp: tuple[float, float] | None = None,
    targeted: bool = False,
    random_start: float | None = None,
    **_: object,
) -> AttackResult:
    """
    PGD attack with boxed constraints and projected L∞ ball around ``input_tensor``.

    Parameters
    ----------
    model
        Differentiable victim emitting logits matching ``true_label``.
    input_tensor
        Clean batch tensor ``(N, C, H, W)``.
    true_label
        ``(N,)`` label tensor (ground-truth or targets depending on ``targeted``).

    Other parameters match standard PGD nomenclature; ``epsilon`` is the L∞ radius.

    Returns
    -------
    AttackResult
        Adversarial examples and per-sample L₂ norms :math:`\\|x' - x\\|_2`.
    """
    ce = criterion or nn.CrossEntropyLoss()

    if alpha is None:

        alpha = epsilon / float(max(steps / 4, 1))

    ori = input_tensor.detach()
    clamp_lo, clamp_hi = clamp if clamp is not None else (-float("inf"), float("inf"))

    x_adv = ori.clone()

    if random_start is not None:

        eta = torch.empty_like(x_adv).uniform_(-random_start, random_start)
        eta = torch.clamp(eta, -epsilon, epsilon)
        x_adv = ori + eta
        if clamp is not None:
            x_adv = torch.clamp(x_adv, clamp_lo, clamp_hi)

    lbl = true_label.long()

    for _ in range(steps):

        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = ce(logits, lbl)
        grad = torch.autograd.grad(loss, x_adv)[0]
        signed = -grad.sign() if targeted else grad.sign()

        x_adv = x_adv.detach() + alpha * signed
        x_adv = torch.max(torch.min(x_adv, ori + epsilon), ori - epsilon)
        if clamp is not None:

            x_adv = torch.clamp(x_adv, clamp_lo, clamp_hi)

    l2_mag = _batch_l2_perturbation(ori, x_adv.detach())

    return AttackResult(adversarial_examples=x_adv.detach(), perturbation_magnitude_l2=l2_mag)

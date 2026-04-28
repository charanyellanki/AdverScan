"""
Fast Gradient Sign Method (FGSM) against differentiable victims (e.g. CIFAR-10 ResNet-18).

Returns adversarial tensors plus measured L₂ perturbation magnitudes ``||x'-x||_2``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from adverscan.attacks.result import AttackResult


def _batch_l2_perturbation(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    diff = (adversarial - clean).view(clean.shape[0], -1)
    return torch.linalg.vector_norm(diff, ord=2, dim=1)


def fgsm(
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: torch.Tensor,
    *,
    epsilon: float,
    criterion: Callable[..., torch.Tensor] | None = None,
    clamp: tuple[float, float] | None = None,
    targeted: bool = False,
    random_start: float | None = None,
    **_: object,
) -> AttackResult:
    """
    FGSM perturbation maximizing (untargeted) cross-entropy w.r.t. ``true_label``.

    Parameters
    ----------
    model
        Victim network ``(x) -> logits`` (e.g. ResNet-18 on CIFAR-10 tensors).
    input_tensor
        Batch ``(N, C, H, W)``; differentiable copy is constructed internally for gradients.
    true_label
        Long tensor ``(N,)`` of class indices aligned with supervised loss evaluation.
    epsilon
        L∞ budget in pixel space (matching ``input_tensor`` coordinate range).
    criterion
        Defaults to reduction-mean CE.
    clamp
        Optional ``[low, high]`` clip after perturbation (e.g. normalized CIFAR range).
    targeted
        If ``True``, ``true_label`` encodes attacker targets (optimize toward targets).
    random_start
        Ignored (API parity with PGD callers).

    Returns
    -------
    AttackResult
        Adversarial batch and ``(N,)`` L₂ norms measuring perturbation magnitude.
    """
    del random_start

    ce = criterion or nn.CrossEntropyLoss()

    imgs = input_tensor.clone().detach().requires_grad_(True)
    logits = model(imgs)

    loss = ce(logits, true_label.long())

    grads = torch.autograd.grad(loss, imgs)[0]
    delta = grads.sign()

    adv = imgs + (epsilon * (-delta) if targeted else epsilon * delta)

    adv = adv.detach()

    if clamp is not None:

        lo, hi = clamp
        adv = torch.clamp(adv, lo, hi)

    l2_mag = _batch_l2_perturbation(input_tensor.detach(), adv)

    return AttackResult(adversarial_examples=adv, perturbation_magnitude_l2=l2_mag.detach())

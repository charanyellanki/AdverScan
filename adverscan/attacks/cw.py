"""
Carlini-Wagner style L₂ attack minimizing distance subject to hinge loss surrogates.

Implementation uses primal optimization with Adam on additive perturbations; ``epsilon``
shapes perturbation initializer scale (coordinate scale).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adverscan.attacks.result import AttackResult


def _batch_l2_perturbation(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:

    diff = (adversarial - clean).view(clean.shape[0], -1)
    return torch.linalg.vector_norm(diff, ord=2, dim=1)


def cw_attack(
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: torch.Tensor,
    *,
    epsilon: float = 1.0,
    steps: int = 300,
    learning_rate: float = 0.01,
    c: float = 5.0,
    clamp: tuple[float, float] | None = None,
    targeted: bool = False,
    kappa: float = 40.0,
    **_: object,
) -> AttackResult:
    """
    CW-style hinge plus L₂ distance objective minimized with Adam unconstrained deltas.

    Parameters
    ----------
    model
        Victim emitting logits shaped ``(N, classes)``.
    input_tensor
        Clean minibatch aligned with ``dtype`` precision expected by CUDA kernels.
    true_label
        ``(N,)`` tensors enumerating benign labels (or CW targets under ``targeted=True``).
    epsilon
        Scales random initialization of free-form perturbation tensor.
    steps
        Adam iterations.
    learning_rate
        Adam step size.
    c
        Trade-off between margin objectives and L₂ distance.
    clamp
        Elementwise clip after each forward (use domain of ``input_tensor``; CIFAR normalized
        inputs often use wider bounds than ``[0,1]``).
    targeted
        ``True`` optimizes cross-entropy toward ``true_label`` classes; otherwise hinge on margins.
    kappa
        Margin constant inside hinge for untargeted attacks.

    Returns
    -------
    AttackResult
        Final adversarial batch with ``(N,)`` L₂ norms between clean and adversarial tensors.
    """
    device = input_tensor.device
    batch = input_tensor.size(0)
    lo, hi = clamp if clamp is not None else (float("-inf"), float("inf"))
    imgs = torch.clamp(input_tensor.detach(), lo, hi)

    perturb = torch.randn_like(imgs, device=device) * (epsilon / 255.0)
    perturb.requires_grad_(True)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    opt = torch.optim.Adam([perturb], lr=learning_rate)

    labs = true_label.long()

    for _ in range(steps):

        opt.zero_grad(set_to_none=True)
        x_adv = torch.clamp(imgs + perturb, lo, hi)
        logits = model(x_adv)

        loss_dist = perturb.view(batch, -1).pow(2).sum(dim=1).mean()

        if targeted:

            margin_obj = criterion(logits, labs)

        else:

            one_hot = torch.nn.functional.one_hot(labs, num_classes=logits.size(1)).bool()
            other = logits.masked_fill(one_hot, float("-inf"))
            best_other = other.max(dim=1)[0]
            real = logits.gather(1, labs.unsqueeze(1)).squeeze(1)
            margin_obj = torch.clamp(best_other - real + kappa, min=0.0).mean()

        loss = loss_dist + c * margin_obj
        loss.backward()
        opt.step()

    adv = torch.clamp(imgs + perturb.detach(), lo, hi)
    l2_mag = _batch_l2_perturbation(imgs, adv)

    return AttackResult(adversarial_examples=adv, perturbation_magnitude_l2=l2_mag.detach())

"""
Four-dimensional anomaly features: softmax entropy, margin, gradient L2 norm, MC-dropout agreement.

Victim models are differentiable PyTorch modules (e.g. CIFAR-10 ResNet-18).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 4


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over softmax probabilities ``(N,)``."""
    probs = F.softmax(logits, dim=-1)
    logp = probs.clamp(min=1e-12).log()
    return -(probs * logp).sum(dim=-1)


def softmax_margin(logits: torch.Tensor) -> torch.Tensor:
    """Difference between strongest and runner-up softmax responses ``(N,)``."""
    probs = F.softmax(logits, dim=-1)
    pair = torch.topk(probs, k=2, dim=-1).values

    return pair[:, 0] - pair[:, 1]


def prediction_consistency_dropout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    num_passes: int = 5,
    dropout_p: float = 0.15,
) -> torch.Tensor:
    """
    Agreement rate across stochastic logits after MC dropout (:math:`[0,1]` per sample).

    ``model.eval()`` keeps BN deterministic while logits receive dropout noise analogous to latent dropout.
    """
    model.eval()

    votes: list[torch.Tensor] = []

    x = input_tensor.detach()

    for _ in range(num_passes):

        logits = model(x)

        perturbed_logits = F.dropout(logits, p=dropout_p, training=True)

        votes.append(perturbed_logits.argmax(dim=-1))

    stack = torch.stack(votes, dim=0)

    mode_vals = torch.mode(stack, dim=0).values

    return (stack == mode_vals.unsqueeze(0)).float().mean(dim=0)


def assemble_extracted_features(
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: torch.Tensor | None = None,
    *,
    num_dropout_passes: int = 5,
    dropout_logits_p: float = 0.15,
) -> torch.Tensor:
    """
    Stack ``[entropy, top1−top2 margin, ‖∇_x CE‖_2 , dropout-agreement]`` rows ``(N, 4)``.

    Labels default to per-row argmax logits when ``true_label`` is ``None`` (evaluation-time estimate).
    """
    model.eval()

    x_req = input_tensor.detach().clone().requires_grad_(True)

    logits_geo = model(x_req)

    ent_stack = softmax_entropy(logits_geo)

    margin_stack = softmax_margin(logits_geo)

    supervised = (
        true_label.detach().long() if true_label is not None else logits_geo.detach().argmax(dim=-1).long()
    )

    loss_vector = nn.CrossEntropyLoss(reduction="mean")(logits_geo, supervised)

    grad_tensor = torch.autograd.grad(loss_vector, x_req, create_graph=False)[0]

    grads_vec = torch.linalg.vector_norm(grad_tensor.view(grad_tensor.shape[0], -1), ord=2, dim=1)

    consistent_scores = prediction_consistency_dropout(
        model,
        input_tensor,
        num_passes=num_dropout_passes,
        dropout_p=dropout_logits_p,
    )

    stacked = torch.stack([ent_stack, margin_stack, grads_vec.detach(), consistent_scores.detach()], dim=1)

    return stacked


class FeatureExtractor(nn.Module):
    """Adapter turning ``assemble_extracted_features`` into a Torch module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def extract(
        self,
        input_tensor: torch.Tensor,
        true_label: torch.Tensor | None = None,
        *,
        num_dropout_passes: int = 5,
        dropout_logits_p: float = 0.15,
        device_cpu: bool = False,
    ) -> torch.Tensor:

        feats = assemble_extracted_features(
            self.model,
            input_tensor,
            true_label,
            num_dropout_passes=num_dropout_passes,
            dropout_logits_p=dropout_logits_p,
        )

        if device_cpu:

            return feats.detach().cpu()

        return feats

    def forward(self, input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:

        return self.extract(input_tensor, **kwargs)


def perturbation_norm(
    clean: torch.Tensor,
    adulterated: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:

    diff = (adulterated - clean).view(clean.shape[0], -1)

    return torch.linalg.vector_norm(diff, ord=p, dim=1)


__all__ = [
    "FEATURE_DIM",
    "FeatureExtractor",
    "assemble_extracted_features",
    "perturbation_norm",
    "prediction_consistency_dropout",
    "softmax_entropy",
    "softmax_margin",
]

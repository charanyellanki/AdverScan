"""
Operating-point tooling: ROC/PR thresholds and false-positive budget sweeps.

These utilities summarize how detector logits translate into deployment trade-offs without
embedding themselves inside the training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_curve


@dataclass(slots=True)
class RocSummary:
    """Container for ROC evaluation outputs."""

    fpr: NDArray[np.floating]
    tpr: NDArray[np.floating]
    roc_thresholds: NDArray[np.floating]


def build_roc(
    labels: NDArray[np.integer],
    scores: NDArray[np.floating],
) -> RocSummary:
    """
    Convenience wrapper emitting ``RocSummary`` envelopes.

    Positive class ``1`` should correspond to adversarial samples.
    """
    curve = roc_curve(labels, scores)
    fpr_np: NDArray[np.floating] = curve[0].astype(np.float32, copy=False)
    tpr_np: NDArray[np.floating] = curve[1].astype(np.float32, copy=False)
    thr_np: NDArray[np.floating] = curve[2].astype(np.float32, copy=False)
    return RocSummary(fpr=fpr_np, tpr=tpr_np, roc_thresholds=thr_np)


@dataclass(slots=True)
class PrecisionRecallSweep:
    """Precision-recall envelopes + threshold ladders."""

    precision: NDArray[np.floating]
    recall: NDArray[np.floating]
    pr_thresholds: NDArray[np.floating]


def build_precision_recall_sweep(labels: NDArray[np.integer], scores: NDArray[np.floating]) -> PrecisionRecallSweep:
    """Produce precision/recall grids using sklearn internals."""
    p, r, thr = precision_recall_curve(labels, scores)
    return PrecisionRecallSweep(
        precision=p.astype(np.float32, copy=False),
        recall=r.astype(np.float32, copy=False),
        pr_thresholds=thr.astype(np.float32, copy=False),
    )


def fp_budget_analysis(
    fpr_curve: Iterable[float],
    target_fp_rates: Iterable[float],
) -> dict[float, tuple[float | None, float]]:
    """
    Map desired false-positive rates to nearest ROC operating points.

    Parameters
    ----------
    fpr_curve
        ROC false-positive probabilities monotonic increasing iterable.
    target_fp_rates
        Iterable of desired budgets ``tau`` with ``tau`` in ``[0,1]``.

    Returns
    -------
    dict
        Lookup ``desired_fp_rate`` -> ``(matched_fpr_observed | None, index_of_closest)``
    """

    fps = np.asarray(list(fpr_curve), dtype=np.float32)
    lookups: dict[float, tuple[float | None, float]] = {}
    if fps.size == 0:
        for target in target_fp_rates:
            lookups[float(target)] = (None, float(np.nan))
        return lookups

    ordered = fps.copy()
    for target in target_fp_rates:
        pos = np.searchsorted(np.sort(ordered), target, side="left")
        if pos >= ordered.size:
            lookups[float(target)] = (None, float("nan"))
            continue

        distances = np.abs(ordered - target)
        argmin_idx = np.argmin(distances)
        matched = float(ordered[argmin_idx])
        lookups[float(target)] = (matched, float(argmin_idx))
    return lookups


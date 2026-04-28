"""Evaluation primitives for benchmarking adversarial detection coverage."""

from adverscan.evaluation.harness import EvaluationHarness, HarnessConfig
from adverscan.evaluation.threshold_analysis import (
    PrecisionRecallSweep,
    RocSummary,
    build_precision_recall_sweep,
    build_roc,
    fp_budget_analysis,
)

__all__ = [
    "EvaluationHarness",
    "HarnessConfig",
    "PrecisionRecallSweep",
    "RocSummary",
    "build_precision_recall_sweep",
    "build_roc",
    "fp_budget_analysis",
]

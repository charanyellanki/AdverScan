"""Detector primitives: feature embeddings, supervised baselines, heuristics."""

from adverscan.detector.baselines import (
    DetectorProto,
    FeatureSqueezeDetector,
    LIDDetector,

    median_smoothing_torch,

    pairwise_local_intrinsic_dimensionality,
)

from adverscan.detector.feature_extractor import (
    FEATURE_DIM,

    FeatureExtractor,

    assemble_extracted_features,

    perturbation_norm,

    softmax_entropy,

    softmax_margin,
)

from adverscan.detector.model import AdversarialDetector, TrainingReport, train_and_select_classifier

__all__ = [
    "FEATURE_DIM",
    "AdversarialDetector",
    "DetectorProto",

    "FeatureExtractor",

    "FeatureSqueezeDetector",

    "LIDDetector",

    "assemble_extracted_features",

    "median_smoothing_torch",

    "pairwise_local_intrinsic_dimensionality",

    "perturbation_norm",

    "softmax_entropy",

    "softmax_margin",

    "train_and_select_classifier",

    "TrainingReport",

]

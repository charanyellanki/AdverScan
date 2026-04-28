"""
REST API for adversarial-vs-clean detection on CIFAR-style tensors.

Serve with::

    uvicorn adverscan.api.main:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from numpy.typing import NDArray

from pydantic import BaseModel, Field

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adverscan.attacks import build_pretrained_cifar10_resnet18

from adverscan.detector.feature_extractor import FEATURE_DIM, assemble_extracted_features

from adverscan.detector.model import AdversarialDetector


app = FastAPI(title="AdverScan Detector API", version="0.1.0")


class InferenceTensor(BaseModel):
    pixels: list[float]
    shape: tuple[int, int, int]


class InferenceRequest(BaseModel):
    tensor: InferenceTensor


class DetectionResponse(BaseModel):
    adversarial_probability: float = Field(ge=0.0, le=1.0)
    feature_dim: int
    softmax_entropy: float
    softmax_margin: float
    gradient_l2: float
    prediction_consistency: float


def _bootstrap_pipeline() -> Pipeline:
    rng = np.random.default_rng(seed=4242)

    phantom_x = rng.normal(size=(256, FEATURE_DIM)).astype(np.float32)

    phantom_y = rng.integers(low=0, high=2, size=phantom_x.shape[0], dtype=np.int64)

    pipe = Pipeline(
        steps=[("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=13))]
    )

    pipe.fit(phantom_x, phantom_y)

    return pipe


@lru_cache(maxsize=1)
def detector_bundle() -> AdversarialDetector:
    artifact = os.getenv("ADVERSCAN_DETECTOR_ARTIFACT", "artifacts/detector.joblib")

    if os.path.isfile(artifact):

        return AdversarialDetector.load(artifact)

    clf = _bootstrap_pipeline()

    detector_holder = AdversarialDetector(backend="logistic_regression", pipeline=clf, val_metrics={}, train_metrics={})

    os.makedirs(os.path.dirname(artifact) or ".", exist_ok=True)

    detector_holder.save(artifact)

    return detector_holder


@lru_cache(maxsize=1)
def cached_victim() -> nn.Module:
    victim_module, succeeded = build_pretrained_cifar10_resnet18()

    if not succeeded:

        import warnings

        warnings.warn("Continuing with randomly initialized ResNet-18 stub (no pretrained CIFAR-10 weights resolved).")

    return victim_module.eval()


@app.post("/predict", response_model=DetectionResponse)
def predict(req: InferenceRequest) -> DetectionResponse:

    accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stacked_tensor = torch.tensor(req.tensor.pixels, dtype=torch.float32, device=accelerator).view(*req.tensor.shape)

    victim_net_nn = cached_victim().to(accelerator)

    feats_torch_stack = assemble_extracted_features(victim_net_nn, stacked_tensor.unsqueeze(0))

    feats_row_np: NDArray[np.floating] = feats_torch_stack.detach().cpu().numpy()[0]

    probs_scalar_vector = detector_bundle().predict_adversarial_score(np.expand_dims(feats_row_np, axis=0))

    det_prob_f = float(probs_scalar_vector[0])

    return DetectionResponse(
        adversarial_probability=det_prob_f,
        feature_dim=FEATURE_DIM,
        softmax_entropy=float(feats_row_np[0]),
        softmax_margin=float(feats_row_np[1]),
        gradient_l2=float(feats_row_np[2]),
        prediction_consistency=float(feats_row_np[3]),

    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


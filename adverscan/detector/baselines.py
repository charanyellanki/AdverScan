"""Baseline anomaly detectors via median squeezing plus LID embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


TorchEmbeddingFn = Callable[[torch.Tensor], NDArray[np.floating]]
DetScoreFunc = Callable[[torch.Tensor], NDArray[np.floating]]

def median_smoothing_torch(images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:

    """

    Per-channel reflective median filter (spatial robust smoothing).

    """
    if kernel_size % 2 != 1 or kernel_size < 3:

        raise ValueError("`kernel_size` odd >=3.")

    pad = kernel_size // 2
    padded = F.pad(images, pad=(pad, pad, pad, pad), mode="reflect")

    batch_sz, chan, ht, wt = images.shape

    smooth = padded.new_zeros((batch_sz, chan, ht, wt))

    radius = kernel_size

    for ch in range(chan):

        unf = torch.nn.functional.unfold(padded[:, ch : ch + 1], kernel_size=radius)

        medians = unf.median(dim=1)[0]

        resh = medians.view(batch_sz, ht, wt)

        smooth[:, ch] = resh

    return smooth


class DetectorProto(ABC):

    """Abstract baseline producing scalar heuristic scores convertible to heuristic probabilities."""

    @abstractmethod
    def scores(self, inputs: torch.Tensor) -> NDArray[np.floating]:

        ...

    def probabilities(self, inputs: torch.Tensor) -> NDArray[np.floating]:

        raw = np.asarray(self.scores(inputs), dtype=np.float32).ravel()

        lo, hi = float(raw.min()), float(raw.max())

        span = hi - lo

        if span < 1e-9:

            return np.ones_like(raw, dtype=np.float32)

        return ((raw - lo) / span).astype(np.float32, copy=False)


class FeatureSqueezeDetector(DetectorProto):

    """

    Median-smoothing squeeze plus KL divergence on logits.

    """
    def __init__(self, victim: nn.Module, *, morph_func: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:

        super().__init__()
        self.victim = victim.eval()

        self.morph = morph_func or median_smoothing_torch

    @torch.no_grad()

    def _kl_between_logits(self, first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:

        p = torch.clamp(torch.softmax(first, dim=-1), min=epsilon, max=1.0 - epsilon)

        q = torch.clamp(torch.softmax(second, dim=-1), min=epsilon, max=1.0 - epsilon)

        return torch.sum(q * torch.log(q / torch.clamp(p, min=epsilon)), dim=-1)

    @torch.no_grad()

    def scores_from_tensors(self, images: torch.Tensor) -> torch.Tensor:

        primary = self.victim(images)

        smoothed_pixels = self.morph(images)

        secondary = self.victim(smoothed_pixels)

        return self._kl_between_logits(primary, secondary)

    def scores(self, images: torch.Tensor) -> NDArray[np.floating]:

        tensors = self.scores_from_tensors(images.detach())

        return tensors.detach().cpu().numpy().astype(np.float32, copy=False)


def pairwise_local_intrinsic_dimensionality(
    embeddings: NDArray[np.floating],

    *,
    neighbours_window: int = 20,

) -> NDArray[np.floating]:
    """

    Lightweight LID proxy exploiting kNN radii logarithmic ratios Ma et al. (2018) style.

    """
    feats = embeddings.astype(np.float64, copy=False)

    k_use = max(8, neighbours_window)

    k_use = min(k_use, feats.shape[0] - 1)

    if k_use <= 1:

        return np.ones(feats.shape[0], dtype=np.float32)

    searcher = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")

    searcher.fit(feats)

    distances_nn, _idx = searcher.kneighbors(feats, return_distance=True)

    distances_trim = distances_nn[:, 1:]  # remove self-distances (~0).

    numerator = distances_trim[:, :-1]

    denom = distances_trim[:, -1].reshape(distances_trim.shape[0], 1)

    ratios_logged = numerator / denom.clip(min=1e-9)

    lid_approx = -(np.nanmean(np.log(ratios_logged + 1e-12), axis=1))

    return lid_approx.astype(np.float32, copy=False)


class LIDDetector(DetectorProto):

    """

    Local intrinsic dimensionality baseline applied to ``victim.embed`` when available.

    """

    def __init__(self, victim: nn.Module | None = None, *, embedding_fn: TorchEmbeddingFn | None = None) -> None:

        self.victim = victim

        self.embedding_fn = embedding_fn

    def _embed(self, batch: torch.Tensor) -> NDArray[np.floating]:

        if self.embedding_fn is not None:

            return self.embedding_fn(batch)

        if self.victim is None:

            raise ValueError("`victim` or `embedding_fn` required.")

        net = self.victim

        mapper = getattr(net, "embed", None)

        with torch.no_grad():

            feats = mapper(batch.detach()) if callable(mapper) else torch.flatten(net(batch), start_dim=1)

        return feats.detach().cpu().numpy().astype(np.float32)

    def scores(self, images: torch.Tensor) -> NDArray[np.floating]:
        feats = self._embed(images)
        lids = pairwise_local_intrinsic_dimensionality(feats, neighbours_window=max(len(feats) // 16, 8))
        return lids


__all__ = [
    "DetectorProto",

    "FeatureSqueezeDetector",
    "LIDDetector",

    "median_smoothing_torch",
    "pairwise_local_intrinsic_dimensionality",

]

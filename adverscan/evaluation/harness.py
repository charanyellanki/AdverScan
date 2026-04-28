"""Sweep attacks and quantify detector KPIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support

from adverscan.attacks import AttackName, get_attack
from adverscan.detector.feature_extractor import FEATURE_DIM, FeatureExtractor
from adverscan.detector.model import AdversarialDetector

VictimFactory = Callable[[], nn.Module]
FeatureExtractorCtor = Callable[[nn.Module], FeatureExtractor]


@dataclass(slots=True)
class HarnessConfig:
    epsilon_values: Sequence[float]
    attacks: Sequence[AttackName]
    clamp: tuple[float, float]
    max_batches: int | None = None
    pgd_random_start: float | None = None


class EvaluationHarness:
    """Apply attacks yielding ``AttackResult`` and evaluate sklearn detectors."""

    def __init__(
        self,
        *,
        victim_ctor: VictimFactory,
        feature_extractor_ctor: FeatureExtractorCtor,
        detector: AdversarialDetector,
        config: HarnessConfig,
        device: torch.device | None = None,
    ) -> None:
        self.victim_ctor = victim_ctor
        self.feature_extractor_ctor = feature_extractor_ctor
        self.detector = detector
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_numpy(self, extractor: FeatureExtractor, images: torch.Tensor, labels: torch.Tensor) -> NDArray[np.float32]:
        cpu_tensor = extractor.extract(images, true_label=labels, device_cpu=True)
        feats = cpu_tensor.detach().numpy().astype(np.float32)
        assert feats.shape[1] == FEATURE_DIM
        return feats

    def run_loader(
        self, *, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[tuple[str, float], dict[str, float]]:
        out: dict[tuple[str, float], dict[str, float]] = {}

        for attack_name in self.config.attacks:
            atk = get_attack(attack_name)
            victim_net = self.victim_ctor().to(self.device).eval()
            extractor = self.feature_extractor_ctor(victim_net)

            for eps in self.config.epsilon_values:
                neg_parts: list[NDArray[np.float32]] = []
                adv_parts: list[NDArray[np.float32]] = []
                batches = 0

                for imgs, lbls in loader:
                    imgs_dev = imgs.to(self.device)
                    lbl_dev = lbls.to(self.device)
                    extras: dict[str, object] = {"clamp": self.config.clamp}
                    if attack_name == "pgd" and self.config.pgd_random_start is not None:
                        extras["random_start"] = self.config.pgd_random_start

                    atk_out = atk(victim_net, imgs_dev, lbl_dev, epsilon=float(eps), **extras)
                    adv_x = atk_out.adversarial_examples.detach()

                    neg_parts.append(self._to_numpy(extractor, imgs_dev, lbl_dev))
                    adv_parts.append(self._to_numpy(extractor, adv_x, lbl_dev))

                    batches += 1
                    if self.config.max_batches is not None and batches >= self.config.max_batches:
                        break

                clean_mat = np.concatenate(neg_parts, axis=0)
                adv_mat = np.concatenate(adv_parts, axis=0)

                feats = np.concatenate([clean_mat, adv_mat], axis=0)
                y_true = np.concatenate(
                    (np.zeros(clean_mat.shape[0], dtype=np.int64), np.ones(adv_mat.shape[0], dtype=np.int64))
                )

                scores = np.asarray(self.detector.predict_adversarial_score(feats), dtype=np.float32)
                preds = scores >= 0.5

                pr = precision_recall_fscore_support(
                    y_true, preds.astype(np.int64), average="binary"
                )

                out[(attack_name, float(eps))] = {
                    "precision": float(pr[0]),
                    "recall": float(pr[1]),
                    "f1": float(pr[2]),
                    "positive_adv_fraction": float(np.mean(y_true.astype(np.float32))),
                    "samples_total": float(y_true.size),
                    "features_dim_aligned": float(FEATURE_DIM),
                }

        return out

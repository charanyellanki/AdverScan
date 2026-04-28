"""AdverScan Streamlit frontend (Hugging Face Spaces friendly).

Local run::
    PYTHONPATH=. streamlit run app.py --server.port 7860 --server.headless true
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adverscan.attacks import build_pretrained_cifar10_resnet18
from adverscan.detector.feature_extractor import FEATURE_DIM, assemble_extracted_features
from adverscan.detector.model import AdversarialDetector
from adverscan.ui.preprocess import pil_to_cifar_tensor


def _bootstrap_detector_pipeline() -> Pipeline:
    rng = np.random.default_rng(seed=4242)
    phantom_x = rng.normal(size=(256, FEATURE_DIM)).astype(np.float32)
    phantom_y = rng.integers(0, 2, size=phantom_x.shape[0], dtype=np.int64)
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=13)),
        ]
    )
    pipe.fit(phantom_x, phantom_y)
    return pipe


@st.cache_resource(show_spinner="Loading ResNet-18 victim…")
def load_victim() -> tuple[nn.Module, bool]:
    net, ok = build_pretrained_cifar10_resnet18()
    net.eval()
    return net, ok


@st.cache_resource(show_spinner="Loading detector artifact…")
def load_detector() -> AdversarialDetector:
    path = Path(os.getenv("ADVERSCAN_DETECTOR_ARTIFACT", Path("artifacts") / "detector.joblib")).expanduser()
    if path.is_file():
        return AdversarialDetector.load(str(path))

    clf = _bootstrap_detector_pipeline()
    detector = AdversarialDetector(
        backend="logistic_regression",
        pipeline=clf,
        train_metrics={},
        val_metrics={},
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        detector.save(str(path))
    except OSError:
        warnings.warn(f"Detector not persisted ({path}); read-only volume?", RuntimeWarning)
    return detector


def main() -> None:
    st.set_page_config(
        page_title="AdverScan",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("AdverScan · adversarial input screening")

    victim, pretrained_ok = load_victim()
    detector = load_detector()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    victim = victim.to(device)

    with st.sidebar:
        st.markdown("### Model status")
        st.write("**Torch device:**", "`" + str(device) + "`")
        st.write(
            "**Victim weights:**",
            "checkpoint resolved" if pretrained_ok else "random init ⚠️ (set env or upload)",
        )
        st.caption(
            "Optional checkpoint env vars: "
            "`ADVERSCAN_VICTIM_CHECKPOINT`, `ADVERSCAN_CIFAR10_RESNET18`."
        )
        tau = st.slider("Flag if **P(adv) ≥ τ**", 0.0, 1.0, value=0.5, step=0.01)

    st.markdown(
        "Upload an RGB image. Inputs are resized to **32×32** with "
        "**CIFAR-10 normalization** (mean/std 0.5), fed to ResNet-18 features, "
        "then distilled into softmax geometry, sensitivity, and stochastic logits signals."
    )

    upload = st.file_uploader(
        "Image file",
        type=["png", "jpg", "jpeg", "webp"],
        help="Arbitrary raster; preprocessing matches CIFAR-style training defaults.",
    )
    if upload is None:
        st.info("Choose an image to compute features and classifier scores.")
        return

    pil = Image.open(upload).convert("RGB")
    c1, c2 = st.columns((1, 1))
    with c1:
        st.image(pil, caption="Uploaded (RGB)", use_container_width=True)

    tensor_chw = pil_to_cifar_tensor(pil).to(device=device, dtype=torch.float32)
    feats = assemble_extracted_features(victim, tensor_chw.unsqueeze(0))
    feats_np = feats.detach().cpu().numpy()

    probs = detector.predict_adversarial_score(feas_np.astype(np.float32, copy=False))
    p_adv = float(probs[0])

    labels = [
        "Softmax entropy",
        "Top-1 minus top-2 margin",
        "Input-gradient L2 (CE)",
        "MC-dropout logits agreement",
    ]

    feat_row = feats_np.reshape(-1)
    with c2:
        st.subheader("Detector output")
        st.metric("P(adversarial)", f"{p_adv:.4f}")
        flagged = bool(p_adv >= tau)
        if flagged:
            st.success("Above threshold τ — escalate or review.")
        else:
            st.info("Below threshold — provisional clean verdict.")
        rows = [{"Feature": labels[i], "Value": float(feat_row[i])} for i in range(min(len(labels), feat_row.size))]
        st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

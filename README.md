# AdverScan

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/)

**AdverScan** detects **adversarial inputs** aimed at deep image classifiers. It wires together a **differentiable victim network** (CIFAR-10–style **ResNet-18** plus FGSM/PGD/C&W generators), **four interpretable behavioral features** per sample (softmax geometry, gradient sensitivity, stochastic consistency), **supervised sklearn heads** (optional XGBoost), and **heuristic baselines** (median feature squeezing, LID proxies). A **FastAPI** service exposes probabilistic **`/predict`** scores; reproducible containers target **CPU PyTorch** stacks for portable deployment experiments.

---

## Why this matters

Evasion adversaries perturb inputs so neural classifiers confidently pick the wrong class while perturbations remain small or perceptually tame. Operational systems—from fleet perception stacks to moderated media pipelines—assume mostly benign distributions. Screening or audit workflows therefore need detectors that summarize **behavior under attack** faster than brute-force label inspection. AdverScan packages that posture as a toolchain: perturbation drivers, discriminative embeddings, calibrated scores, threshold analytics, and baselines grounded in canonical literature motifs.

---

## Architecture (conceptual pipeline)

Scores flow **Input → Features → Detector → Probabilities → Threshold / Budget**. Operational teams map detector probabilities to ROC/PR-supported cut points honoring **false-positive budgets**.

```
                      +--------------------------+
                      | Batch (N,C,H,W) tensor x|
                      +-----------+--------------+
                                  |
                                  v
            +--------------------------------------------+
            |            Feature extractor               |
            |  • softmax entropy H(p)                    |
            |  • top1 − top2 margin                      |
            |  • ‖∇x CE‖2 (estimated input sensitivity) |
            |  • MC dropout agreement (K passes)       |
            +-------------------+-----------------------+
                                |
                                |  vector ∈ R^4 per row
                                v
            +--------------------------------------------+
            |        Binary detector (sklearn pipeline) |
            |  StandardScaler + Logistic / RF / XGBoost|
            |  label 0 = clean, label 1 = adversarial |
            +-------------------+-----------------------+
                                |
                                v
                     P(adversarial | x) ∈ [0,1]
                                |
                                v
           +--------------------------------------------+
           |   Threshold τ  & FP-budget tooling        |
           |   PR/ROC curves, α-FPR anchored operating   |
           |   points (see evaluation/threshold_analysis)
           +--------------------------------------------+
```

---

## Evaluation results (representative placeholders)

Illustrative **macro-F1** ordering consistent with empirical surveys (multi-step/search attacks often degrade separation vs. single-step). Replace with **`evaluation/harness.py`** metrics once datasets and checkpoints are fixed.

| Attack | Budget (illus.) | Macro-F1* | Precision* | Recall* |
|--------|-----------------|----------|-------------|---------|
| **FGSM** | L∞ (8/255-equiv.) | **0.91** | **0.90** | **0.92** |
| **PGD** | L∞, PGD-10 style | **0.87** | **0.86** | **0.88** |
| **C&W** | L2 optimization | **0.83** | **0.84** | **0.82** |

\*Rounded illustrative figures—not peer-reviewed measurements from this repo.

---

## Threshold analysis and FP budget methodology

Operating points trade **accepted false positives on clean inputs** versus **coverage on adversarial traffic**. Practical deployments often constrain **fraction of clean uploads sent to escalation** (“FP budget”). The workflow mirrored in `evaluation/threshold_analysis.py` is:

1. Collect detector scores plus ground-truth clean vs adversarial indicators on representative validation mixtures.
2. Build **ROC** (`build_roc`) and **precision–recall ladders** (`build_precision_recall_sweep`).
3. Tie an FP budget **`α`** to the ROC **false-positive-rate** axis; `fp_budget_analysis()` maps nominal α anchors to discrete ROC coordinates from finite validation draws.
4. Propagate thresholds per attack regime—attacks shift score distributions asymmetrically.

This keeps engineering guardrails intuitive: SOC review bandwidth, SLA on manual review queues, regulatory false-alarm quotas.

---

## Baseline comparison (placeholder AUROC cues)

Normalized comparative scores hypothetical for exposition; rerun with **`FeatureSqueezeDetector`** and **`LIDDetector`** on your splits.

| Method | Mechanism sketch | Typical adv-focused AUROC* | Complexity |
|--------|-----------------|----------------------------|-----------|
| **AdverScan (supervised features + sklearn)** | Learned fusion of softmax/gradient/dropout stats | **0.93** | Medium training cost |
| **Feature squeezing (median smoothing)** | Input median filter + KL divergence on logits | **0.74** | Low latency |
| **LID heuristic (embedding k-NN ratios)** | Neighbor-distance dispersion of latent vectors | **0.71** | Medium latency |

\*Illustrative; validate per dataset.

---

## Quickstart

### Local environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r adverscan/requirements.txt
export PYTHONPATH="$(pwd)"

python -c "import adverscan; print(adverscan.__version__)"
```

Train detectors offline via **`train_and_select_classifier`** in `adverscan/detector/model.py` using numpy matrices from **`assemble_extracted_features`**.

### API (development)

```bash
export PYTHONPATH="$(pwd)"

uvicorn adverscan.api.main:app --reload --host 0.0.0.0 --port 8000

curl -fsS http://127.0.0.1:8000/healthz
```

### Docker & Compose

```bash
docker compose build
docker compose up
curl -fsS http://127.0.0.1:8000/healthz
```

Artifacts and checkpoints configure through environment variables (see **`adverscan/.env.example`**, especially **`ADVERSCAN_VICTIM_CHECKPOINT`**, **`ADVERSCAN_CIFAR10_RESNET18`**, **`ADVERSCAN_DETECTOR_ARTIFACT`**).

### Streamlit UI (Hugging Face Spaces)

The repository root includes **`app.py`** for a **Streamlit** dashboard: image upload, CIFAR-style preprocessing, four extracted features, and **`P(adversarial)`** with an adjustable threshold.

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
streamlit run app.py --server.port 7860 --server.headless true
```

HF’s **[Streamlit Docker template](https://huggingface.co/spaces/streamlit/streamlit-template-space)** runs **`streamlit run src/streamlit_app.py`** and originally only **`COPY`s `src/` + `requirements.txt`**. This repo adds **`src/streamlit_app.py`** (delegates to **`app.py`**) plus **[`Dockerfile.space`](Dockerfile.space)** — copy **`adverscan/`**, **`app.py`**, installs full **`requirements.txt`**. Replace the Space’s generated **`Dockerfile`** with **`Dockerfile.space`** (or merge the `COPY` lines). Details: [`docs/HUGGING_FACE_SPACES.md`](docs/HUGGING_FACE_SPACES.md).

---

## Repository layout

| Path | Responsibility |
|------|----------------|
| [`adverscan/attacks/`](adverscan/attacks/) | FGSM, PGD, CW, **`AttackRunner`**, `AttackResult` |
| [`adverscan/attacks/resnet_cifar10.py`](adverscan/attacks/resnet_cifar10.py) | CIFAR ResNet-18 + checkpoint helpers |
| [`adverscan/detector/feature_extractor.py`](adverscan/detector/feature_extractor.py) | Four-dimensional behavioral embedding |
| [`adverscan/detector/model.py`](adverscan/detector/model.py) | Detector training + **`train_and_select_classifier`** |
| [`adverscan/detector/baselines.py`](adverscan/detector/baselines.py) | Feature squeezing + LID surrogates |
| [`adverscan/api/main.py`](adverscan/api/main.py) | FastAPI `/predict` & `/healthz` |
| [`app.py`](app.py) | Streamlit UI (local); logic reused by Spaces |
| [`src/streamlit_app.py`](src/streamlit_app.py) | HF Docker template entry → calls `app.main()` |
| [`Dockerfile.space`](Dockerfile.space) | **Spaces**: replace generated Dockerfile (COPY `adverscan/`, deps, Streamlit 8501) |
| [`docs/HUGGING_FACE_SPACES.md`](docs/HUGGING_FACE_SPACES.md) | Hugging Face Spaces deployment guide |
| [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) | FastAPI CPU image + compose (**not** the HF Streamlit Dockerfile) |

---

## Tech stack badges (summary)

| Layer | Choices |
|-------|---------|
| Language | ![Python](https://img.shields.io/badge/python-3.11-blue.svg) core package |
| Surrogate model | ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white) ResNet attacks + differentiable forward |
| Detector | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) + optional **`xgboost`** |
| Service | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) + **`uvicorn`**; optional ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) UI |
| Ops | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) multi-stage Dockerfile + **`docker-compose.yml`** |

---

## Package documentation

Structural notes for downstream packaging live under [`adverscan/README.md`](adverscan/README.md) (companion module overview).

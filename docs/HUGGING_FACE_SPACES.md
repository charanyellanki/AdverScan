# Hugging Face Spaces (Streamlit Docker template)

Your Space was created from **[streamlit/streamlit-template-space](https://huggingface.co/spaces/streamlit/streamlit-template-space)**. That template:

- Runs **`streamlit run src/streamlit_app.py`** (port **8501**).
- By default **`COPY`s only `requirements.txt` and `src/` into the image** — it **does not** include the **`adverscan/`** Python package unless you extend the Dockerfile.

AdverScan needs the **`adverscan/`** directory, **`app.py`**, and the **root [`requirements.txt`](../requirements.txt)** (torch, sklearn, etc.). Follow the steps below so the Space build matches this repository.

---

## 1. Add this repo’s code to the Space

Push **this full repository** to the Space (replace the template-only files), **or** merge manually:

| Must exist in Space repo root |
|------------------------------|
| [`adverscan/`](../adverscan/) package |
| [`app.py`](../app.py) |
| [`src/streamlit_app.py`](../src/streamlit_app.py) (thin entry that calls [`app.main()`](../app.py)) |
| Root [`requirements.txt`](../requirements.txt) (`-r adverscan/requirements.txt` + `streamlit`) |
| Optional: [`.streamlit/config.toml`](../.streamlit/config.toml) |

---

## 2. Replace the Space `Dockerfile`

The template Dockerfile only installs `pandas` / `altair` / `streamlit`.

**Overwrite** `Dockerfile` in the Space with the contents of **[`Dockerfile.space`](../Dockerfile.space)** in this repo (same directory as this doc), or duplicate its `COPY` / `pip install` / `ENTRYPOINT`.

Key differences from the default template:

- **`COPY adverscan`** and **`COPY app.py`** so imports work (`PYTHONPATH=/app`).
- **Python 3.11** (recommended for stable PyTorch wheels; the template uses 3.13 which can be picky).
- Keeps **`ENTRYPOINT`** aligned with **`src/streamlit_app.py`** and port **8501**.

---

## 3. Replace `requirements.txt`

Replace the Space’s minimal `requirements.txt` with **[`requirements.txt` at repo root](../requirements.txt)** (it includes **`adverscan/requirements.txt`** plus Streamlit).

---

## 4. How `src/streamlit_app.py` fits in

[Hugging Face’s template expects](https://huggingface.co/spaces/streamlit/streamlit-template-space/blob/main/Dockerfile) `src/streamlit_app.py`.

This repo wires that file to **`app.py`** so you keep a **single** UI implementation:

- **Local**: `PYTHONPATH=. streamlit run app.py`
- **Space (Docker)** : `streamlit run src/streamlit_app.py` (adds repo root to `sys.path`, then runs `main()` from `app.py`)

---

## 5. Environment variables

Under **Space → Settings → Variables and secrets**:

| Variable | Meaning |
|----------|---------|
| `ADVERSCAN_VICTIM_CHECKPOINT` or `ADVERSCAN_CIFAR10_RESNET18` | ResNet-18 CIFAR-10 checkpoint (optional; else random-init warning). |
| `ADVERSCAN_DETECTOR_ARTIFACT` | **`joblib`** detector artifact (optional; else bootstrap logistic on phantom data). |

---

## Non-Docker Spaces (pure Streamlit SDK)

If Hugging Face offers **Streamlit** as a SDK without Docker, it often runs **`app.py`** at the repo root automatically. Then you **do not** need `src/streamlit_app.py` or a custom Dockerfile; still ship **`adverscan/`** and **`requirements.txt`**.

---

## README YAML (optional)

````markdown
---
title: AdverScan
emoji: 👁️
sdk: docker
pinned: false
---
````

(Customize `sdk` / metadata to match your Space settings.)

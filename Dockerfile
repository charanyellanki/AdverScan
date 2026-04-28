# Multi-stage image: slim runtime with PyTorch CPU wheels.
#
#   docker compose up --build
#   docker build -t adverscan-api .
#

ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY adverscan/requirements.docker.txt /build/requirements.docker.txt

RUN python -m venv /opt/vendor \
    && /opt/vendor/bin/pip install --upgrade pip setuptools wheel \
    && /opt/vendor/bin/pip install torch torchvision \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && /opt/vendor/bin/pip install --no-cache-dir -r requirements.docker.txt

# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/vendor/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/vendor /opt/vendor

COPY adverscan /app/adverscan

RUN useradd --create-home --uid 65532 app \
    && mkdir -p /data/artifacts \
    && chown -R app:app /app /data

ENV PYTHONPATH=/app:/opt/vendor/lib/python3.11/site-packages \
    ADVERSCAN_DETECTOR_ARTIFACT=/data/artifacts/detector.joblib

USER app

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "adverscan.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

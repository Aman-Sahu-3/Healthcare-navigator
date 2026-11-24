# syntax=docker/dockerfile:1.7
#
# Healthcare Knowledge Navigator — API image
#   * multi-stage: build tools and caches stay out of the final image
#   * CPU-only PyTorch (the default wheel bundles ~2 GB of CUDA we don't need)
#   * embedding model downloaded at BUILD time -> fast, offline startup
#   * runs as a non-root user

ARG PYTHON_VERSION=3.12

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
WORKDIR /build

# 1) CPU PyTorch first, so the dependency install below doesn't pull the CUDA build.
ARG TORCH_VERSION=2.15.0
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

# 2) Dependencies only (a stub package), so code edits don't re-install everything.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install . \
    && pip uninstall -y healthcare-navigator

# 3) Bake the embedding model into the image.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}')"

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    ENVIRONMENT=production \
    CHROMA_DIR=/data/chroma \
    RAW_DOCS_DIR=/data/raw \
    EVAL_DIR=/app/eval

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/raw /data/chroma \
    && chown -R appuser:appuser /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf

WORKDIR /app
COPY --chown=appuser:appuser app ./app

USER 10001
EXPOSE 8000

# Generous start period: loading PyTorch + the model takes ~20-40 s on small machines.
# hadolint ignore=DL3025
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)" || exit 1

# One worker: each worker loads its own copy of the model (~300 MB RAM).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

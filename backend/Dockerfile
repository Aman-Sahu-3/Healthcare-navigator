# syntax=docker/dockerfile:1.7
#
# Healthcare Knowledge Navigator — API image
#   * multi-stage: build tools and caches stay out of the final image
#   * ONNX embeddings via FastEmbed: no PyTorch, fits 512 MB hosting tiers
#   * embedding model downloaded at BUILD time -> fast, offline startup
#   * runs as a non-root user; listens on $PORT (Render/Cloud Run) or 8000

ARG PYTHON_VERSION=3.12

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv
WORKDIR /build

# Dependencies only (a stub package), so code edits don't re-install everything.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install . \
    && pip uninstall -y healthcare-navigator

# Bake the ONNX embedding model (~67 MB) into the image.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/opt/fastembed')"

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_CACHE_DIR=/opt/fastembed \
    ENVIRONMENT=production \
    CHROMA_DIR=/data/chroma \
    RAW_DOCS_DIR=/data/raw \
    EVAL_DIR=/app/eval \
    PORT=8000

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/raw /data/chroma \
    && chown -R appuser:appuser /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/fastembed /opt/fastembed

WORKDIR /app
COPY --chown=appuser:appuser app ./app

USER 10001
EXPOSE 8000

# Liveness only — cheap, never touches the index.
# hadolint ignore=DL3025
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health/live', timeout=4)" || exit 1

# One worker: each worker holds its own copy of the model and index in memory.
# Shell form so $PORT expands; exec so uvicorn receives stop signals directly.
# --no-access-log: our middleware already logs each request (as JSON, with request ID).
# hadolint ignore=DL3025
CMD exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1 --no-access-log

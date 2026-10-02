# Production image: FastAPI service + one model bundle. asl/lab is not shipped.
#
#   uv run dvc pull artifacts/model        # or: uv run dvc repro
#   docker build -f docker/api.Dockerfile -t asl-api .
#   docker run -p 8000:8000 asl-api
#
# The model is baked in so an image tag pins one model version; build with
# --build-arg MODEL_DIR=<bundle dir> to package a different bundle.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.8.17

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG UV_VERSION
RUN pip install --no-cache-dir uv==${UV_VERSION}
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --extra api --no-install-project

FROM python:${PYTHON_VERSION}-slim-bookworm
ARG MODEL_DIR=artifacts/model
# Headless OpenCV needs no system libraries, so no apt-get layer.
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY asl/__init__.py asl/__init__.py
COPY asl/prod asl/prod
COPY ${MODEL_DIR} /app/model
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    ASL_MODEL_DIR=/app/model
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"
# One worker per container; scale with replicas (Kubernetes) rather than processes.
CMD ["uvicorn", "asl.prod.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

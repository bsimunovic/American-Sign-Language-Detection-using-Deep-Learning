# Experiment environment: uv + DVC + TensorFlow. The repository is mounted, not
# copied, so DVC sees the git history and cache:
#
#   docker build -f docker/lab.Dockerfile -t asl-lab .
#   docker run --rm -it -v "$PWD":/workspace asl-lab dvc repro
#   docker run --rm -it -v "$PWD":/workspace asl-lab dvc exp run -S model.name=legacy_cnn
#
# For NVIDIA GPUs run with --gpus all after adding tensorflow[and-cuda] to the lab extra.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm
ARG UV_VERSION=0.8.17
RUN pip install --no-cache-dir uv==${UV_VERSION}
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONPATH=/workspace \
    TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace
WORKDIR /workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --extra lab --extra api --group notebooks --no-install-project
# The asl-lab console script lives in the project; expose it via a tiny shim.
RUN printf '#!/bin/sh\nexec python -m asl.lab.cli "$@"\n' > /opt/venv/bin/asl-lab && chmod +x /opt/venv/bin/asl-lab
CMD ["dvc", "repro"]

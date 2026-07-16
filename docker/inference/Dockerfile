# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/shaft/.venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/shaft/.venv \
    PATH=/opt/shaft/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    ninja-build \
    pkg-config \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/shaft

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN uv sync --frozen --no-dev \
    --extra serve \
    --extra qwen-next \
    --extra gpu \
    --extra gpu-kernels

COPY docker/inference/start-vllm.sh /usr/local/bin/shaft-start-vllm
COPY docker/inference/contract-smoke.py /usr/local/bin/shaft-contract-smoke
RUN chmod +x /usr/local/bin/shaft-start-vllm /usr/local/bin/shaft-contract-smoke

EXPOSE 8000

CMD ["shaft-start-vllm"]

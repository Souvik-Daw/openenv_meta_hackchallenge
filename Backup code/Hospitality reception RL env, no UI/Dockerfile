# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# IMPORTANT: Docker build context must be the project ROOT (where pyproject.toml lives):
#   docker build -f Dockerfile -t healthcare-scheduling:latest .
#   docker run --rm -p 8000:8000 -e GROQ_API_KEY=gsk_... healthcare-scheduling:latest
#
# To run the baseline evaluation inside the container:
#   docker run --rm -e GROQ_API_KEY=gsk_... healthcare-scheduling:latest \
#       python run_baseline.py --oracle
#
# For Hugging Face Spaces: openenv push uploads this file to the Space repo root.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=standalone
ARG ENV_NAME=healthcare_scheduling

# Copy the full project root into /app/env
COPY . /app/env
WORKDIR /app/env

# ── Ensure uv is available ─────────────────────────────────────────────────
RUN if ! command -v uv > /dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# ── Install dependencies (cached layer) ────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# ── Install the project itself ─────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv and project files from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env       /app/env

# HF Spaces runs as user 1000 (non-root)
RUN chown -R 1000:1000 /app

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONPATH: /app/env is the package root
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start the FastAPI server
CMD ["uvicorn", "healthcare_scheduling.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/env"]

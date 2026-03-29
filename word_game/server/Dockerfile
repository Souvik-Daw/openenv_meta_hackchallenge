# syntax=docker/dockerfile:1
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# IMPORTANT: Docker build context must be the word_game/ project ROOT (one level above server/):
#   docker build -f server/Dockerfile -t word_game-env:latest .
# This ensures pyproject.toml, uv.lock, models.py, and the full package are available.
#
# For Hugging Face Spaces: openenv push uploads this file to the Space repo root.
# HF Spaces runs containers as user 1000 (non-root) so ownership is set accordingly.

# Allow overriding the base image via openenv push --base-image ...
# Default: standard Python slim so HF Spaces can always pull it without auth.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=standalone
ARG ENV_NAME=word_game

# Copy the full word_game project root into /app/env.
# Build context = word_game/ directory (contains pyproject.toml, uv.lock, models.py, server/).
COPY . /app/env

# Work from the project root where pyproject.toml and uv.lock live.
WORKDIR /app/env

# Ensure uv is available.
RUN if ! command -v uv > /dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies only (not the project) -- cached separately for faster rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Install the project itself into the venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# -- Runtime stage -----------------------------------------------------------
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment and package from builder.
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env       /app/env

# HF Spaces runs as user 1000 (non-root) -- ensure files are readable.
RUN chown -R 1000:1000 /app

# Activate the venv.
ENV PATH="/app/.venv/bin:$PATH"

# /app/env IS the word_game package root (pyproject.toml: package-dir = {"word_game" = "."}).
# PYTHONPATH must include /app (the parent) so `import word_game` resolves correctly.
ENV PYTHONPATH="/app:$PYTHONPATH"

# Declare the port the server listens on (matches README.md app_port: 8000).
EXPOSE 8000

# Health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn using the fully-qualified installed module path.
# Format required by uvicorn: module_path:asgi_app_object
# word_game.server.app:app matches pyproject.toml package layout.
CMD ["uvicorn", "word_game.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]

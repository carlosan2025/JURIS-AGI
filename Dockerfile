# JURIS-AGI Docker Image
# Supports both CPU and GPU (CUDA) modes

# =============================================================================
# Base stage with Python and common dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --upgrade pip && \
    pip install -e ".[api]"

# =============================================================================
# CPU-only image
# =============================================================================
FROM base as cpu

# Install CPU-only PyTorch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Set device to CPU
ENV DEVICE=cpu

# Create non-root user
RUN useradd -m -u 1000 juris
USER juris

# Default command
CMD ["python", "-m", "juris_agi.api.server"]

# =============================================================================
# CUDA/GPU image
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as cuda-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with CUDA PyTorch
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[api]"

# Set device to auto (will detect CUDA)
ENV DEVICE=auto

# Create non-root user
RUN useradd -m -u 1000 juris
USER juris

# Default command
CMD ["python", "-m", "juris_agi.api.server"]

# =============================================================================
# API server image (CPU)
# =============================================================================
FROM cpu as api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "juris_agi.api.server"]

# =============================================================================
# Worker image (GPU)
# =============================================================================
FROM cuda-base as worker

# Workers don't expose ports but need health check via Redis
ENV HEALTH_CHECK_INTERVAL=30

CMD ["python", "-m", "juris_agi.api.worker"]

# =============================================================================
# Default target is the CPU API image
# =============================================================================
FROM api

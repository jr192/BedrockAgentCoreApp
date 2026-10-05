# ==============================================================================
# Multi-Stage Production Dockerfile for Bedrock AgentCore Multi-Agent System
# Architecture: linux/arm64 (AWS Graviton) / linux/amd64 compatible
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.12-slim-trixie AS builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /build

# Create virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy package dependency definition
COPY pyproject.toml .

# Install dependencies into virtual environment
RUN uv pip install --no-cache -r pyproject.toml

# ------------------------------------------------------------------------------
# Stage 2: Final Minimal & Hardened Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim-trixie AS runner

# Runtime environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

# Security: Create non-root user and group
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY --chown=appuser:appuser . /app

# Switch to unprivileged non-root user for enterprise container security
USER appuser

# Expose internal HTTP runtime port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', 8080)); s.close()" || exit 1

# Runtime entrypoint
ENTRYPOINT ["python", "main.py"]

# ABOUTME: Multi-stage Dockerfile for red-teaming runner with uv package manager
# ABOUTME: Optimized for production with minimal final image size

# Stage 1: Build dependencies
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim as builder

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-cache --no-dev

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

# Install uv for runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install required system packages
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash runner

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=runner:runner . .

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/logs && \
    chown -R runner:runner /app/data /app/logs

# Switch to non-root user
USER runner

# Ensure virtual environment is used
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "apps.runner.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
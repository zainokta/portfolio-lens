# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_NO_CACHE=1 \
    PATH="/root/.local/bin:$PATH"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file and install packages
COPY pyproject.toml ./
RUN uv sync --no-dev

# Clean up build dependencies in same layer
RUN apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime image
FROM python:3.12-slim

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-create-home -r -g appuser appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser .env ./
COPY --chown=appuser:appuser prompt.txt ./

# Switch to the non-root user
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

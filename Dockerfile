# Stage 1: Build dependencies
FROM debian:bookworm-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_NO_CACHE=1 \
    PATH="/root/.local/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Copy dependency file and install packages
COPY pyproject.toml ./
RUN uv sync

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_NO_CACHE=1 \
    PATH="/root/.local/bin:$PATH"

# Install only necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application source files
COPY . .

# Copy required data files (even though they're in .gitignore)
COPY backfill.json ./
COPY prompt.txt ./
COPY .env ./

# Run backfill script to initialize database
RUN python script/backfill.py

# Expose FastAPI default port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

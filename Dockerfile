# StockM v1.0 - Phase 9, Lesson 13
# Production Docker image for the Prediction API
# ==============================================
# Multi-stage: build deps in a fat stage, copy to a slim final image to minimize
# attack surface + image size. Runs as a non-root user (security best practice).

# --- Stage 1: builder (install deps) ---
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for numpy/torch/scikit-learn wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python deps (reuse the project's requirements; torch CPU for serving).
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install uvicorn[standard]

# --- Stage 2: runtime (slim) ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from the builder.
COPY --from=builder /install /usr/local

# Copy the application + models + data needed for inference.
# (In CI, models/ and data/prepared/ would be mounted or fetched from a model
# store; here we bake them in for a self-contained image.)
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/prepared/ ./data/prepared/

# Non-root user (security: a container compromise shouldn't run as root).
RUN useradd -m -u 1000 stockm && chown -R stockm:stockm /app
USER stockm

ENV PYTHONPATH=/app/src
ENV STOCKM_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check: orchestrator polls /health to know the container is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn with N workers (override via env). The app is api.main:app.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8000} --workers ${API_WORKERS:-4}"]

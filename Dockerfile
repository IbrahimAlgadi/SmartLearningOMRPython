# ─────────────────────────────────────────────────────────────────────────────
#  OMR Auto-Marking Service — Dockerfile
#  Multi-stage build: slim Python 3.11 base, non-root user, HEALTHCHECK.
#
#  Build:
#    docker build -t omr-api:latest .
#
#  Run:
#    docker run --rm -p 8000:8000 \
#      -e OMR_API_KEY=changeme \
#      -e OMR_DEBUG_ARTIFACTS=0 \
#      -v $(pwd)/web_runs:/app/web_runs \
#      -v $(pwd)/bubble_classifier_v3.onnx:/app/bubble_classifier_v3.onnx:ro \
#      omr-api:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile opencv-python wheels (if not pre-built)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Create a non-root user and group for the application
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser

# Copy application code (excludes items listed in .dockerignore)
COPY --chown=appuser:appgroup . .

# Ensure the run data directory exists and is owned by the app user
RUN mkdir -p /app/web_runs && chown appuser:appgroup /app/web_runs

# Switch to non-root user before running anything
USER appuser

# Expose application port
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker / kubernetes liveness probe — pings /healthz (no auth required).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# ── Default command ───────────────────────────────────────────────────────────
# 2 workers: tune -w to (2*CPU_COUNT)+1 for your target instance.
# --timeout 60   : kill a worker that takes > 60 s without responding.
# --graceful-timeout 30 : wait 30 s for in-flight requests on SIGTERM.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "webapp.main:app"]

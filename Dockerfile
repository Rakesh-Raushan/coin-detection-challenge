FROM python:3.10-slim

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /code

# Copy dependency files
COPY uv.lock pyproject.toml ./

# Install from lock file to system Python
RUN uv export --frozen --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

# Model file to copy (must match default in app/core/config.py)
ARG MODEL_FILE=yolov8n-coin-finetuned.pt

# Copy model directory - the app will handle missing models gracefully at runtime
RUN mkdir -p ./artifacts/models
COPY artifacts/models/ ./artifacts/models/
COPY app/ ./app/

# Create data directory and set permissions
RUN mkdir -p data/uploads && chown -R appuser:appuser /code

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

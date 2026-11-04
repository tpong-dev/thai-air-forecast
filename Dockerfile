# Thai Air Forecast API (CPU only, LightGBM; no torch in the image).
#   docker build -t thai-air-forecast .
#   docker run -p 8000:8000 thai-air-forecast
# models/ and reports/ are baked in. Without saved models the API serves a labelled persistence fallback.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    THAIAIR_MODELS_DIR=/app/models \
    THAIAIR_DATA_DIR=/app/data \
    THAIAIR_THREADS=2

WORKDIR /app

# libgomp is needed by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first for layer caching. Core install only: the deep (torch) extra is not served.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .

COPY app/ app/
COPY models/ models/
COPY reports/ reports/

# Non-root user; data/ is writable for the HTTP response cache.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# FastAPI backend + Laya model.
# The Laya weights (~1.7 GB) are NOT baked into the image. They download on first start
# into /models, which docker-compose mounts as a volume so they're kept between restarts.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    USE_TF=0 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# 1. CPU-only PyTorch first. The default Linux wheel bundles CUDA (~2.5 GB extra) we don't use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

# 2. Everything else. Copied before the code so Docker caches this layer between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. The app, run as a non-root user.
COPY app ./app
RUN useradd --create-home appuser && mkdir -p /models /data && chown appuser /models /data
USER appuser

EXPOSE 8000
# Laya takes 30-180 s to load at startup, hence the long start period.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One worker: each worker would load its own ~2 GB copy of the model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

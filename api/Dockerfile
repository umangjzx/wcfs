# VayuCast API
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY config ./config
COPY ingest ./ingest
COPY aqi ./aqi
COPY features ./features
COPY models ./models
COPY api ./api

RUN pip install --upgrade pip && pip install -e ".[model,api]"

# Bundled model + demo snapshot make the container demoable with no network.
COPY models/registry ./models/registry
COPY demo/snapshot ./data/snapshots

EXPOSE 8000
# Boot from the bundled snapshot; refresh in the background once online.
ENV VAYUCAST_BOOT_INGEST=0 VAYUCAST_REFRESH_MIN=60
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

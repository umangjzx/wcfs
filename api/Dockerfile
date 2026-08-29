# VayuCast API
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgomp1: LightGBM's OpenMP runtime (missing from python:*-slim)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config ./config
COPY ingest ./ingest
COPY aqi ./aqi
COPY features ./features
COPY models ./models
COPY api ./api

# editable so config.settings.REPO_ROOT resolves to /app (where the model lands)
RUN pip install --upgrade pip && pip install -e ".[model,api,postgres]"

# Ship the trained model so the container can forecast as soon as it has live data.
COPY models/registry ./models/registry

EXPOSE 8000
# Live-data only: ingest real CPCB/GFS/FIRMS on boot, then hourly. No bundled snapshot.
ENV VAYUCAST_BOOT_INGEST=1 VAYUCAST_REFRESH_MIN=60
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

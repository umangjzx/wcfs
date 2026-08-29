"""VayuCast API — FastAPI app for the Delhi-NCR coupled AQI forecast.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.endpoints import router
from api.services.pipeline import STATE, load_snapshot, refresh

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vayucast.api")

_REFRESH_MIN = int(os.environ.get("VAYUCAST_REFRESH_MIN", "60"))
_INGEST_ON_BOOT = os.environ.get("VAYUCAST_BOOT_INGEST", "1") != "0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if load_snapshot():
        log.info("loaded forecast snapshot from disk (stale until first refresh)")

    try:
        from api.services.store import enabled, init_schema
        if enabled():
            log.info("postgres mirror %s", "ready" if init_schema() else "configured (schema pending)")
    except Exception as exc:  # noqa: BLE001
        log.warning("postgres mirror unavailable: %s", exc)

    def _boot():
        log.info("initial refresh starting (ingest=%s)", _INGEST_ON_BOOT)
        log.info("initial refresh: %s", refresh(do_ingest=_INGEST_ON_BOOT))

    threading.Thread(target=_boot, daemon=True).start()

    sched = None
    try:
        from api.scheduler import shutdown, start
        sched = start(_REFRESH_MIN)
    except Exception as exc:  # noqa: BLE001
        log.warning("scheduler not started: %s", exc)
    yield
    if sched:
        shutdown()


app = FastAPI(title="VayuCast API", version="0.1.0",
              description="72-hour coupled meteorology–chemistry AQI forecast for Delhi NCR",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VAYUCAST_CORS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "VayuCast API",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/stations", "/api/forecast/{id}", "/api/grid",
                      "/api/drivers/{id}", "/api/fires", "/api/alerts", "/api/model-card"],
        "model": STATE.model_name,
        "last_refresh": STATE.last_refresh,
    }

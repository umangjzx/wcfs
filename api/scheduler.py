"""Hourly refresh of the forecast pipeline (APScheduler)."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from api.services.pipeline import refresh

log = logging.getLogger("vayucast.scheduler")
_scheduler: BackgroundScheduler | None = None


def _job() -> None:
    log.info("scheduled refresh starting")
    status = refresh(do_ingest=True)
    log.info("scheduled refresh done: %s", status)


def start(interval_minutes: int = 60) -> BackgroundScheduler:
    global _scheduler
    if _scheduler:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(_job, "interval", minutes=interval_minutes, id="refresh",
                       next_run_time=None, max_instances=1, coalesce=True)
    _scheduler.start()
    log.info("scheduler started (every %d min)", interval_minutes)
    return _scheduler


def shutdown() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None

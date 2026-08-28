---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 13
  completed_plans: 3
  percent: 23
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-28)

**Core value:** A trustworthy, explainable 72-hour Delhi-NCR AQI forecast that visibly accounts for the inversion ↔ aerosol feedback loop.
**Current focus:** Phase 3 — AQI + coupled feature engineering

## Current Position

Phase: 3 of 8 (AQI + coupled feature engineering)
Plan: 1 of 2 (03-01: CPCB AQI module — already built in Phase 1; 03-02: ISI, stubble vector, feedback, builder)
Status: Ready to build 03-02
Last activity: 2026-08-28 — Phase 2 complete. Ingestion live-verified: CPCB 420 obs/65 stations, GFS +77h/65 stations, ERA5 reanalysis 65 stations. FIRMS wired, needs map key.

Progress: [██░░░░░░░░] 23%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: —

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1. Scaffold | 1/1 | Complete |
| 2. Data ingestion | 2/2 | Complete |

## Accumulated Context

### Decisions

- ERA5 history + GFS forecast via **Open-Meteo JSON** (keyless, cross-platform); `cdsapi` optional. Both confirmed to serve `boundary_layer_height` + 1000/925/850 hPa temps.
- Merged planned `era5.py` + `gfs.py` into one `ingest/weather.py` (`fetch_reanalysis` / `fetch_forecast`) — same API, DRY.
- Station registry **regenerated from the live data.gov.in CPCB feed** (`ingest/stations_build.py`): 65 NCR stations, authoritative names + coords (`coords_verified: true`). id scheme `DEL-…` etc.
- data.gov.in key provided by user (was labelled `MARKET_API_KEY`) — verified against the CPCB resource, wired as `DATA_GOV_IN_API_KEY` in `.env`.
- Wind decomposed to u/v with meteorological "FROM" convention in `ingest/weather.py`.

### Pending Todos

- **FIRMS_MAP_KEY** still needed (free, instant) — stubble-plume feature depends on it. Without it `ingest.firms` degrades to snapshot/empty.
- **OpenAQ v3** ground-truth historical path is a TODO in `ingest/cpcb.fetch_history` — currently returns Open-Meteo CAMS model output (fine to build/demo; swap for OpenAQ when key present for real training targets).
- `WEATHER_API_KEY` + `OPENROUTER_*` stored in `.env`, not wired in. OpenRouter = candidate for LLM-generated advisories later (billing — ask first).

### Blockers/Concerns

- Training-target quality: CAMS proxy vs CPCB ground truth — resolve before Phase 4 training if OpenAQ key arrives.

## Session Continuity

Last session: 2026-08-28
Stopped at: Phase 2 complete (commit pending). Next: Phase 3 / 03-02 — features/inversion.py, features/stubble.py, features/feedback.py, features/build.py.
Resume file: None

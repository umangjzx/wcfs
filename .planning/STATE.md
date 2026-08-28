---
gsd_state_version: '1.0'
status: shippable
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 13
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-28)

**Core value:** A trustworthy, explainable 72-hour Delhi-NCR AQI forecast that visibly accounts for the inversion ↔ aerosol feedback loop.
**Current focus:** All 8 phases done — polish / stretch items remain (TFT, quantile calibration, per-pollutant AQI).

## Current Position

Phase: 8 of 8 complete. End-to-end system runs.
Status: Shippable. Pushed to github.com/umangjzx/wcfs.
Last activity: 2026-08-29 — Phases 5 (WRF-Chem artifact), 6 (API), 7 (dashboard), 8 (packaging) landed. Dashboard verified in-browser against the live API. Model retrained on real OpenAQ winter 2025-26 data.

Progress: [██████████] 100%

## Performance Metrics

| Phase | Status |
|-------|--------|
| 1 Scaffold | Complete |
| 2 Data ingestion | Complete |
| 3 Coupled features | Complete |
| 4 Model + backtest | Complete |
| 5 WRF-Chem artifact | Complete |
| 6 FastAPI service | Complete |
| 7 React/MapLibre dashboard | Complete |
| 8 Packaging | Complete |

## Backtest (real CPCB winter 2025-26, 2-fold walk-forward, 492k forecasts)

MAE 39.9 vs persistence 54.0 vs climatology 62.8 · bias −2.0 · positive skill every horizon 2h+ ·
Very Poor (AQI≥301) POD 0.72 / FAR 0.33 / CSI 0.53 · Severe still weak (CSI 0.10) · P10–P90 coverage 64%.

## Accumulated Context

### Decisions

- Historical CPCB targets: **OpenAQ S3 open-data archive** (keyless, 56/65 stations for winter 2025-26). `ingest.openaq --start … --end …`.
- Historical met: Open-Meteo ERA5 archive (surface + BLH); pressure-level 925/850 hPa unavailable keyless for old dates → ISI runs 3/4 components, stubble transport uses Ekman-veered 10 m wind. Serving path (GFS) is full fidelity. `era5_arco.py` is a VM upgrade path.
- API routes consolidated in one `api/routes/endpoints.py` (not 8 files).
- Dashboard map uses a **self-contained MapLibre style** (no external basemap) — offline-safe; a minimal style must OMIT the `glyphs` key (not set it undefined). `React.StrictMode` removed (double-invoke churns the imperative map).
- Demo resilience: `demo/snapshot/` committed; API boots from it with no keys/network.

### Pending Todos / stretch

- Vite dev-HMR occasionally leaves the map blank after many hot reloads; a fresh dev server / production build is clean. Harden the map lifecycle (recreate on detached container) if it bites.
- 04-02 stretch: Temporal Fusion Transformer (needs `dl` extra + training time).
- Quantile calibration is in-sample → coverage 64%; add conformal calibration on a holdout.
- Severe-event detection weak → class weighting or a dedicated Severe classifier.
- AQI from PM2.5 only → forecast PM10/NO2/O3 too for a true multi-pollutant AQI.
- `runpy` warning on `python -m features.build` — cosmetic.
- OpenRouter key in `.env`, unused (LLM advisories — billing, ask first).

### Blockers/Concerns

None blocking. External-service friction (Open-Meteo rate limits on bulk pulls, OpenAQ location-id fragmentation, ARCO-ERA5 broken stores) documented in PROJECT.md.

## Session Continuity

Last session: 2026-08-29
Stopped at: All phases complete; repo pushed to github.com/umangjzx/wcfs.
Resume file: None

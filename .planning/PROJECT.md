# VayuCast — Air Pollution–Weather Coupled Forecasting System (Delhi NCR)

## What This Is

VayuCast is a 72-hour AQI forecasting system for the Delhi National Capital Region that
explicitly models the two-way meteorology–chemistry feedback the standard forecasts ignore:
temperature inversions trapping PM2.5 near the surface, and dense aerosol loading in turn
suppressing boundary-layer height. It ingests live CPCB air-quality, ERA5/GFS meteorology,
and NASA FIRMS stubble-fire data, serves hourly probabilistic forecasts from a fast ML
emulator, and presents them on a React/MapLibre dashboard for the public and disaster
managers. Built for SIH 2026 Problem Statement 26082 (MoES → NCMRWF).

## Core Value

A trustworthy, explainable 72-hour AQI forecast for Delhi NCR that visibly accounts for the
inversion ↔ aerosol feedback loop — if only one thing works, it is the forecast number and
the "why" behind it.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] REQ-01: Ingest live + historical CPCB pollutant observations for ~35–40 NCR stations
- [ ] REQ-02: Ingest ERA5 reanalysis (incl. boundary-layer height) and GFS forecast meteorology
- [ ] REQ-03: Ingest NASA FIRMS fire hotspots over Punjab/Haryana/NCR
- [ ] REQ-04: Compute Indian National AQI (CPCB method) from pollutant concentrations
- [ ] REQ-05: Derive coupling features — Inversion Strength Index, stubble-plume transport vector, aerosol→PBL feedback terms
- [ ] REQ-06: Train an ML emulator producing 72h hourly PM2.5/AQI forecasts with P10/P50/P90 bands
- [ ] REQ-07: Backtest with walk-forward CV; beat persistence and climatology on MAE and Severe-onset detection
- [ ] REQ-08: Produce an offline WRF-Chem validation artifact for one historical stubble spike
- [ ] REQ-09: Serve forecasts, drivers, grid, fires, and alerts over a FastAPI service, refreshed hourly
- [ ] REQ-10: React + MapLibre dashboard: NCR map, 72h time slider, per-station forecast, drivers panel, alerts
- [ ] REQ-11: Package with docker-compose; offline snapshot fallback for demo resilience

### Out of Scope

- Live/rolling WRF-Chem serving — too slow and compute-heavy for a hackathon demo; used once offline for validation only
- Pan-India coverage — scoped to Delhi + Gurugram + Noida + Ghaziabad + Faridabad to keep data tractable
- Source-apportionment chemistry beyond stubble vs local — not needed for the forecast + advisory goal
- Mobile native apps — responsive web only

## Context

- Attached research (SIH 2026 PS Research & Selection Guide) rates 26082 the primary pick: live public data, defensible technical story, low prior-art conflict. It prescribes the ML-emulator-with-offline-WRF-Chem-validation strategy adopted here.
- Public data sources verified in that guide: CPCB API (data.gov.in), ERA5 (Copernicus CDS), NASA FIRMS, NOAA GFS. Published Delhi WRF-Chem pipeline: GMD *DSS v1.0* (gmd.copernicus.org/articles/17/2617/2024, FINN + EDGAR + mozbc).
- Team registers its own API keys; ingestion must degrade gracefully to cached snapshots without them.
- Idea submission deadline: 20 September 2026.

## Constraints

- **Tech stack**: Python 3.13 (pandas/xarray/LightGBM/PyTorch/FastAPI), React + Vite + TS + MapLibre GL — team familiarity + hackathon speed
- **Compute**: No HPC — live path must run on a laptop / small VM; WRF-Chem is a one-off prep run
- **Data**: Historical CPCB coverage uneven before 2021 — training focuses on the three most recent complete winters
- **Timeline**: Working demo before the SIH internal round; idea deck by 20 Sep 2026
- **Workflow**: gsd-core phase loop (Discuss → Plan → Execute → Verify → Ship); artifacts in `.planning/`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fast ML emulator (LightGBM → TFT) serves all live forecasts | Full WRF-Chem too slow to refresh live; guide's recommended strategy | — Pending |
| One offline WRF-Chem run for validation only | Proof of physics engagement without HPC on the live path | — Pending |
| Coupling encoded as engineered features (ISI, plume vector, aerosol→PBL) | Lets the emulator learn the feedback the PS calls out; explainable | — Pending |
| GFS for inference-time meteorology, ERA5 for training history | ERA5 is reanalysis (past only); GFS is free forecast data | — Pending |
| React + Vite + MapLibre for the dashboard | Best judge impression; interactive map is core to the demo | — Pending |
| DuckDB + Parquet, no DB server | Hackathon simplicity; optional Postgres/Timescale compose profile | — Pending |
| NCR scoped to 5 cities' CAAQMS stations | Keeps ingestion and station metadata tractable | — Pending |

---
*Last updated: 2026-08-28 at project initialization*

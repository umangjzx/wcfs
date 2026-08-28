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

### Done in the upgrade pass (commits 6f025ff, 965a2a4)

- **Conformal quantile calibration** — `models/conformal.py` (CQR, per-horizon additive
  margins from a held-out tail slice). Replaces the in-sample `interval_k`.
- **Multi-pollutant** — `MultiForecaster` trains PM2.5 + PM10 + NO2, derives the real
  CPCB AQI (max sub-index + dominant pollutant). `--fast` = PM2.5 only for iteration.
- **Feature matrix 167 → 299 cols** — lags to 48 h for 6 pollutants + coupled features,
  rolling std/max, `isi_x_pm25`, `stubble_x_isi`, `pm25_anom`, blh/vent tendencies,
  6h wind steadiness, `hours_since_rain`, rush-hour flags.
- **Rare-event handling** — mild P90-only up-weighting + `event_score()` (~P75) drives
  alert / event decisions instead of the biased median. (First attempt over-weighted
  everything → +38 bias; fixed in 965a2a4.)
- Map lifecycle hardened (recreate on detached container); `features/__init__` lazy
  (runpy warning gone).

### Still stretch

- Full 3-target train is slow locally — use `notebooks/train_colab.ipynb` on a T4.
- Temporal Fusion Transformer (04-02) — the `dl` extra + `models/tft.py` slot.
- OpenRouter key in `.env`, unused (LLM advisories — billing, ask first).

### Blockers/Concerns

None blocking. External-service friction (Open-Meteo rate limits on bulk pulls, OpenAQ location-id fragmentation, ARCO-ERA5 broken stores) documented in PROJECT.md.

## Session Continuity

Last session: 2026-08-29
Stopped at: All phases complete; repo pushed to github.com/umangjzx/wcfs.
Resume file: None

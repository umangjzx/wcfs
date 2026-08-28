---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 13
  completed_plans: 5
  percent: 42
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-28)

**Core value:** A trustworthy, explainable 72-hour Delhi-NCR AQI forecast that visibly accounts for the inversion ↔ aerosol feedback loop.
**Current focus:** Phase 4 — ML forecasting model + backtest (04-01 landing)

## Current Position

Phase: 4 of 8 (ML forecasting model + backtest)
Plan: 04-01 (LightGBM baseline + backtest) — code complete + running on real data
Status: Iterating on baseline metrics; blockers are known and scoped (see below)
Last activity: 2026-08-29 — full pipeline runs end to end on the Oct 2023–Jan 2024 window (obs+met+fires → 173k feature rows → LGBM P10/P50/P90 → walk-forward backtest). Fixed a categorical-encoding bug; added L1 median + persistence blend + interval calibration.

Progress: [████░░░░░░] 42%

## Performance Metrics

| Phase | Plans | Status |
|-------|-------|--------|
| 1. Scaffold | 1/1 | Complete |
| 2. Data ingestion | 2/2 | Complete |
| 3. Features | 2/2 | Complete |
| 4. Model + backtest | ~1/2 | In progress |

## Backtest snapshot (2-fold walk-forward, Dec 2023–Jan 2024, CAMS targets)

Post categorical-bug-fix run: MAE 29.9 (persistence 39.0, climatology 33.9);
**positive skill vs persistence from 6 h onward (+0.24 … +0.41)**; Very Poor CSI 0.32.
Remaining issues → fixes in flight / next:
- systematic under-bias (−21) and Severe-event blindness → **CAMS PM2.5 caps ~225; needs real OpenAQ targets**
- P10–P90 coverage 37% → added interval calibration (`LGBMForecaster.calibrate`)
- 1–3 h worse than persistence → added persistence blend for h ≤ 6

## Accumulated Context

### Decisions

- `models/`: `dataset.make_supervised` (multi-horizon, curated ~55-feature allow-list),
  `baseline_lgbm.LGBMForecaster` (L1 median + P10/P90 quantile boosters, horizon as a
  feature, persistence blend h≤6, calibrated interval multiplier), `backtest.run`
  (expanding walk-forward, gap, vs persistence + hour-of-year climatology).
- CATEGORICAL = station_id/site_type/city/local_hour/local_month/is_weekend/is_stubble_season.
- Training window in use: **CAMS obs + ERA5 met + FIRMS fires, 2023-10-01 … 2024-01-20**
  (173k feature rows). CAMS is the model backbone; OpenAQ real data merges where available.

### Pending Todos

- **Swap in real CPCB targets (OpenAQ S3)** for a recent complete winter, rebuild features,
  re-backtest. `ingest.openaq --start … --end …` (keyless S3 archive; 60/65 stations mapped).
  OpenAQ Indian data is location-id-fragmented → coverage partial per year; stitch or accept.
- LightGBM training is slow on this box (~3 min/fit). Curated feature list already cut it
  ~3×; consider `HistGradientBoostingRegressor` or fewer boost rounds if iteration hurts.
- 04-02: TFT + SHAP + registry.
- OpenRouter key stored, unused (LLM advisories later — billing, ask first).

### Blockers/Concerns

- Historical 925/850 hPa met: documented compromise (ERA5 archive surface + BLH; ISI 3/4
  components; Ekman-veered 10 m transport wind). Serving path is full fidelity. `era5_arco.py`
  / `weather.py --past-days` are VM upgrade paths.

## Session Continuity

Last session: 2026-08-29
Stopped at: backtest re-running with L1 median + blend + calibration. Next: read results, commit 04-01, then OpenAQ real-target swap.
Resume file: None

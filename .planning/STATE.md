---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 13
  completed_plans: 5
  percent: 38
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-28)

**Core value:** A trustworthy, explainable 72-hour Delhi-NCR AQI forecast that visibly accounts for the inversion ↔ aerosol feedback loop.
**Current focus:** Phase 4 — ML forecasting model + backtest

## Current Position

Phase: 4 of 8 (ML forecasting model + backtest)
Plan: 1 of 2 (04-01: LightGBM multi-horizon quantile baseline + backtest harness)
Status: Ready to plan — pending a decision on the historical pressure-level met source (see Blockers)
Last activity: 2026-08-28 — Phase 3 complete. Feature matrix builds end-to-end (23.4k rows x 167 cols for the Nov-2023 test window, ~4 s). ISI diurnal cycle validated (night 0.66 / midday 0.12); corr(self_trapping, PM2.5) = +0.59.

Progress: [████░░░░░░] 38%

## Performance Metrics

| Phase | Plans | Status |
|-------|-------|--------|
| 1. Scaffold | 1/1 | Complete |
| 2. Data ingestion | 2/2 | Complete |
| 3. Features | 2/2 | Complete |

## Accumulated Context

### Decisions

- Feature modules: `features/{inversion,stubble,feedback,calendar_feats,build}.py`. `build.build_matrix` assembles pivot→hourly grid→AQI→ISI→stubble→feedback→calendar→lags/rollings.
- ISI = weighted blend of θ-inversion, PBL shallowness, near-surface calm, radiative-inversion terms; **renormalizes over available components** when pressure-level temps are missing.
- Stubble-plume transport uses 850 hPa wind, **falls back to 10 m wind** when 850 hPa absent.
- `self_trapping = ISI × log1p(PM2.5)` is the headline coupled feature.
- Added `aqi.cpcb_aqi.sub_index_series` (vectorized) for fast AQI over the whole matrix.
- FIRMS: fixed to the API's real **day-range limit of 5**; history now works (Nov 2023 stubble peak clearly recovered, ~230k MW FRP over 15 days). User note: FIRMS quota is 5000 req / 10 min — our worst case is ~180 req.

### Pending Todos

- **OpenAQ key now in `.env`** — implement the OpenAQ v3 ground-truth history path in `ingest/cpcb.fetch_history` (currently returns Open-Meteo CAMS model output). Do before Phase 4 training.
- `WEATHER_API_KEY`, `OPENROUTER_*` stored, unused. OpenRouter = later LLM advisories (billing — ask first).
- `runpy` warning when running `python -m features.build` after `import features` — cosmetic.

### Blockers/Concerns

- **RESOLVED (documented): historical pressure-level met.** Investigated ARCO-ERA5 (both Zarr stores have corrupt `level` coord / non-unique lat-lon index / 2-min opens from Windows) and Open-Meteo `past_days` (pressure levels only ~2 weeks back). Decision: training history = Open-Meteo ERA5 archive (surface + BLH + cloud, always 100%); ISI runs 3/4 components; stubble transport uses an **Ekman-veered 10 m wind** (`_sfc_to_transport_wind`). Serving path unaffected. `ingest/era5_arco.py` + `weather.py --past-days` kept as upgrade paths for a VM. See PROJECT.md decision.

- **NEXT / real unblock for Phase 4: OpenAQ v3 ground-truth history.** Feature validation is currently against the CAMS PM2.5 *model* proxy, so `corr(stubble_load, pm25) ≈ 0` (CAMS has its own burning emissions, doesn't track FIRMS). Real CPCB winter PM2.5 has large stubble-driven spikes. OpenAQ key is in `.env`. Implement `ingest/cpcb.fetch_history` OpenAQ v3 sensor-walk → real targets → re-validate features → then train.

## Session Continuity

Last session: 2026-08-28
Stopped at: Phase 3 complete (commit pending). Next: resolve pressure-level source, then Phase 4 (models/baseline_lgbm.py + backtest).
Resume file: None

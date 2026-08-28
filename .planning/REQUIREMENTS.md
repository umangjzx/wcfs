# Requirements: VayuCast

Scoped requirements for SIH 2026 PS 26082. IDs are referenced by ROADMAP.md phases.

## Functional

| ID | Requirement | Acceptance signal |
|----|-------------|-------------------|
| REQ-01 | Ingest CPCB pollutant observations (PM2.5, PM10, NO2, O3, CO, SO2) for ~35–40 NCR stations, real-time + 3 winters historical | `data/processed/obs.parquet` has ≥30 stations, hourly rows across Oct–Feb of 2021/22/23 |
| REQ-02 | Ingest ERA5 (2m temp/dewpoint, 10m wind, BLH, radiation, cloud, 1000/925/850 hPa temp+wind) and GFS 0–72h forecast met, interpolated to station points | ERA5 season files present; `gfs.py --once` yields 72h hourly met per station |
| REQ-03 | Ingest NASA FIRMS VIIRS/MODIS hotspots over Punjab/Haryana/NCR bbox; aggregate to daily fire clusters with FRP | `firms.py --once` returns non-empty clusters for a November date |
| REQ-04 | Compute Indian National AQI (CPCB sub-index method, correct averaging windows) + category + dominant pollutant | `pytest tests/test_aqi.py` passes against published CPCB worked examples |
| REQ-05 | Coupling features: Inversion Strength Index (0–1 + components), stubble-plume transport vector (scalar load + 2D vector), aerosol→radiation→PBL interaction terms | `features/build.py` output contains all columns; `tests/test_features.py` geometry checks pass |
| REQ-06 | ML emulator: 72h hourly forecast of PM2.5 (primary) + PM10/NO2/O3, quantiles P10/P50/P90, AQI derived | `models/train.py --model lgbm` persists a model; predict returns 72×quantile array |
| REQ-07 | Walk-forward backtest over 3 winters with metrics: MAE/RMSE per horizon, AQI-category accuracy, POD/FAR/CSI for Very Poor (301+) and Severe (401+) onset, skill vs persistence + climatology, quantile calibration | `models/backtest.py` prints table; MAE(24–72h) < persistence, Severe-onset CSI > persistence |
| REQ-08 | Offline WRF-Chem validation: Delhi-NCR nest namelists, FINN+EDGAR+mozbc runbook, notebook comparing simulated surface PM2.5 vs CPCB for one stubble spike | `wrfchem/` has namelists + `pipeline.md` + `validate.ipynb` with a comparison figure |
| REQ-09 | FastAPI service: `/stations`, `/observations`, `/forecast/{id}`, `/grid`, `/drivers/{id}`, `/fires`, `/alerts`, `/model-card`; APScheduler hourly refresh; results cached | `uvicorn api.main:app` up; `curl /api/forecast/<id>` → 72 quantile points; scheduler logs a completed cycle |
| REQ-10 | React + Vite + MapLibre dashboard: NCR AQI choropleth + station markers, now→+72h time slider, per-station forecast chart with band + advisory, drivers panel (ISI gauge, stubble meter, SHAP bars), alerts banner, methodology page | `npm run dev`; all panels render against local API; slider updates map |
| REQ-11 | docker-compose (api + web + scheduler), README with key-registration links + demo script, seeded offline forecast snapshot | `docker compose up` serves dashboard end-to-end; demo works with network disabled |

## Non-functional

| ID | Requirement |
|----|-------------|
| NFR-01 | Live forecast cycle (ingest → features → predict → cache) completes in < 5 min on a laptop |
| NFR-02 | Missing API keys or upstream outages never crash the service — fall back to last good snapshot with a staleness flag |
| NFR-03 | All coupling features are individually inspectable and surfaced in the drivers endpoint (explainability) |
| NFR-04 | `pytest` green; lint (ruff) clean; basic CI on push |
| NFR-05 | Forecasts carry calibrated uncertainty (P10/P50/P90), not just point estimates |

## Verification strategy

End-to-end checks live in ROADMAP.md success criteria and the plan file
(`C:/Users/UMANG JAISWAL N/.claude/plans/peppy-knitting-milner.md`, "Verification" section).

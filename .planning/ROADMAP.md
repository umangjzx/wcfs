# Roadmap: VayuCast

## Overview

Build outward from data to demo. First stand up ingestion for the four public feeds, then
compute AQI and the coupling features that make this forecast "coupled" rather than a plain
time-series model. Train and honestly backtest an ML emulator, produce the offline WRF-Chem
validation artifact for credibility, wrap everything in a FastAPI service with an hourly
refresh, then build the React/MapLibre dashboard and package it for the demo.

## Phases

- [x] **Phase 1: Scaffold & workflow bootstrap** - Repo skeleton, gsd-core planning artifacts, station + settings config
- [x] **Phase 2: Data ingestion** - CPCB, ERA5, GFS, FIRMS ingest to Parquet/DuckDB
- [x] **Phase 3: AQI + coupled feature engineering** - CPCB AQI, Inversion Strength Index, stubble-plume vector, aerosol→PBL feedback
- [ ] **Phase 4: ML forecasting model + backtest** - LightGBM baseline → TFT, walk-forward CV, SHAP
- [ ] **Phase 5: Offline WRF-Chem validation artifact** - Namelists, FINN+EDGAR+mozbc runbook, comparison notebook
- [ ] **Phase 6: Backend API** - FastAPI endpoints + APScheduler hourly refresh + cache
- [ ] **Phase 7: Frontend dashboard** - React + Vite + MapLibre: map, time slider, forecast, drivers, alerts
- [ ] **Phase 8: Integration, packaging, demo polish** - docker-compose, README, offline snapshot, pitch assets

## Phase Details

### Phase 1: Scaffold & workflow bootstrap
**Goal**: A working Python + JS project skeleton with gsd-core planning artifacts and seed config.
**Depends on**: Nothing (first phase)
**Requirements**: REQ-04 (partial — AQI module stub), NFR-04
**Success Criteria** (what must be TRUE):
  1. `pip install -e .` succeeds; `ruff`/`pytest` run (even if trivial)
  2. `.planning/` holds PROJECT, REQUIREMENTS, ROADMAP, STATE, config.json
  3. `config/stations.yaml` lists ~35–40 real CPCB NCR stations with lat/lon, city, site type
  4. `config/settings.py` exposes Delhi centroid, ingest bboxes, plume `tau`, AQI thresholds, data paths
  5. `git` repo initialized with `.gitignore` excluding `data/` and model blobs
**Plans**: 1 plan

Plans:
- [x] 01-01: Repo skeleton, pyproject, config, git init, CI stub

### Phase 2: Data ingestion
**Goal**: One command refreshes all four feeds into typed Parquet tables.
**Depends on**: Phase 1
**Requirements**: REQ-01, REQ-02, REQ-03, NFR-02
**Success Criteria**:
  1. `python -m ingest.cpcb --once` writes tidy observations for ≥30 NCR stations
  2. `python -m ingest.era5 --seasons 2021,2022,2023` downloads and station-interpolates ERA5 fields
  3. `python -m ingest.gfs --once` yields 72h hourly forecast meteorology per station
  4. `python -m ingest.firms --once` returns daily fire clusters with FRP for a November window
  5. `python -m ingest.run_ingest --once` runs the full cycle and degrades gracefully when a key is missing
**Plans**: 2 plans

Plans:
- [x] 02-01: CPCB real-time + historical (OpenAQ/CAMS) + station registry
- [x] 02-02: ERA5 + GFS (ingest/weather.py, Open-Meteo), FIRMS + run_ingest orchestrator

### Phase 3: AQI + coupled feature engineering
**Goal**: A per-station hourly feature matrix that encodes the met–chem coupling.
**Depends on**: Phase 2
**Requirements**: REQ-04, REQ-05, NFR-03
**Success Criteria**:
  1. `pytest tests/test_aqi.py` passes against published CPCB worked examples
  2. `features/inversion.py` returns ISI in [0,1] plus named components
  3. `features/stubble.py` returns incoming-load scalar + 2D plume vector; unit test: fire due upwind → vector points at Delhi
  4. `features/build.py` writes `data/processed/features.parquet` with obs + met + fire + calendar + lags
  5. `notebooks/02_feature_checks` shows ISI and stubble-load rising during known Nov spikes
**Plans**: 2 plans

Plans:
- [x] 03-01: CPCB AQI module + tests (+ vectorized sub_index_series)
- [x] 03-02: ISI, stubble-plume vector, aerosol→PBL feedback, calendar, feature builder

### Phase 4: ML forecasting model + backtest
**Goal**: A persisted emulator that produces calibrated 72h probabilistic AQI forecasts and beats naive baselines.
**Depends on**: Phase 3
**Requirements**: REQ-06, REQ-07, NFR-05
**Success Criteria**:
  1. `python -m models.train --model lgbm` persists a model + feature list + metrics.json
  2. Prediction returns 72 hourly steps with P10/P50/P90 for PM2.5/PM10/NO2/O3 and derived AQI
  3. `python -m models.backtest` prints per-horizon MAE/RMSE, category accuracy, POD/FAR/CSI
  4. MAE(24–72h) < persistence; Severe-onset CSI > persistence; quantile calibration within tolerance
  5. SHAP contributions persisted for the drivers endpoint
**Plans**: 2 plans

Plans:
- [~] 04-01: LightGBM multi-horizon quantile baseline + backtest harness (code done + synthetic-tested; real-data backtest numbers pending — slow training box + OpenAQ target swap)
- [ ] 04-02: TFT model + SHAP + model registry

### Phase 5: Offline WRF-Chem validation artifact
**Goal**: Reproducible evidence the team engaged with the real coupled physics.
**Depends on**: Phase 3 (needs CPCB obs for comparison)
**Requirements**: REQ-08
**Success Criteria**:
  1. `wrfchem/namelist.wps` + `namelist.input` define a Delhi-NCR nest for one stubble-spike event
  2. `wrfchem/pipeline.md` documents Docker image + FINN + EDGAR + mozbc steps, citing GMD DSS v1.0
  3. `wrfchem/validate.ipynb` compares simulated surface PM2.5 vs CPCB (time series + scatter + bias)
  4. Runbook is executable on a single cloud VM; not wired into the live API
**Plans**: 1 plan

Plans:
- [ ] 05-01: Namelists, pipeline runbook, validation notebook

### Phase 6: Backend API
**Goal**: A single service exposes forecasts, drivers, grid, fires, and alerts, refreshed hourly.
**Depends on**: Phase 4
**Requirements**: REQ-09, NFR-01, NFR-02, NFR-03
**Success Criteria**:
  1. `uvicorn api.main:app` serves all eight endpoints with Pydantic-typed responses
  2. `curl /api/forecast/<id>` returns 72 hourly points with quantiles + category + dominant pollutant
  3. `curl /api/drivers/<id>` returns SHAP contributors + ISI components + stubble load + plume vector
  4. APScheduler runs the ingest→features→predict→cache cycle hourly and on startup
  5. Killing upstream network still serves the last good snapshot with a staleness flag
**Plans**: 2 plans

Plans:
- [ ] 06-01: Endpoints, schemas, services, cache
- [ ] 06-02: Scheduler, snapshot fallback, model-card

### Phase 7: Frontend dashboard
**Goal**: An interactive NCR dashboard that tells the coupling story.
**Depends on**: Phase 6
**Requirements**: REQ-10
**Success Criteria**:
  1. MapLibre NCR map shows AQI choropleth + station markers coloured by CPCB category
  2. Time slider now → +72h re-queries and updates the map and selected station
  3. Station panel shows P50 line + P10–P90 band, category badge, health advisory, dominant pollutant
  4. Drivers panel shows ISI gauge + components, stubble-load meter, SHAP contributor bars
  5. Alerts banner shows soonest Severe/Very-Poor crossing with lead time; methodology page renders model-card
**Plans**: 2 plans

Plans:
- [ ] 07-01: App shell, map, station markers, API client
- [ ] 07-02: Forecast chart, drivers panel, alerts, time slider, methodology page

### Phase 8: Integration, packaging, demo polish
**Goal**: One command brings up the whole system; demo is resilient.
**Depends on**: Phase 7
**Requirements**: REQ-11, NFR-04
**Success Criteria**:
  1. `docker compose up` starts api + scheduler + web; dashboard reachable
  2. Full ingest→forecast→UI path works against live feeds
  3. With network disabled, the seeded snapshot still drives a complete demo
  4. README documents key registration, setup, and a scripted demo walkthrough
  5. Pitch assets exist: architecture diagram, coupling one-slide, backtest metrics table, WRF-Chem figure
**Plans**: 1 plan

Plans:
- [ ] 08-01: docker-compose, README, snapshot seeding, pitch assets, CI

## Progress

**Execution Order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
(Phase 5 depends only on Phase 3 and may run in parallel with Phase 4/6 if a VM is available.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffold & workflow bootstrap | 1/1 | Complete | 2026-08-28 |
| 2. Data ingestion | 2/2 | Complete | 2026-08-28 |
| 3. AQI + coupled feature engineering | 2/2 | Complete | 2026-08-28 |
| 4. ML forecasting model + backtest | 0/2 | Not started | - |
| 5. Offline WRF-Chem validation artifact | 0/1 | Not started | - |
| 6. Backend API | 0/2 | Not started | - |
| 7. Frontend dashboard | 0/2 | Not started | - |
| 8. Integration, packaging, demo polish | 0/1 | Not started | - |

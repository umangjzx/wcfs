# VayuCast

**Air Pollution–Weather Coupled Forecasting System — Delhi NCR**
Smart India Hackathon 2026 · Problem Statement **26082** · Ministry of Earth Sciences → NCMRWF · *Disaster Management*

VayuCast produces a **72-hour, hourly, probabilistic AQI forecast** for the Delhi National
Capital Region that explicitly models the two-way meteorology–chemistry feedback standard
forecasts ignore:

- **Inversion → aerosol** — temperature inversions and a shallow planetary boundary layer trap PM2.5 near the surface.
- **Aerosol → inversion** — dense aerosol loading blocks sunlight, cools the surface and further suppresses boundary-layer growth.

The live forecast is served by a fast ML emulator trained on engineered coupling features
(an **Inversion Strength Index** and a **stubble-plume transport vector**). A one-off,
offline **WRF-Chem** run grounds the emulator in the real coupled physics.

## Results (walk-forward backtest, real CPCB winter 2025-26, 492k forecasts)

| | MAE | RMSE | bias |
| --- | --- | --- | --- |
| **VayuCast** | **37.2 µg/m³** | **54.8** | **+4.2** |
| Persistence | 54.0 | 80.5 | +2.0 |
| Hour-of-year climatology | 62.8 | 85.2 | +17.3 |

- Positive skill vs persistence at **every horizon 1 h – 72 h** (+0.02 → +0.40)
- **"Very Poor" (AQI ≥ 301) episode detection: POD 0.96 / FAR 0.44 / CSI 0.55** (persistence CSI 0.43)
- **"Severe" (AQI ≥ 401): POD 0.39 / CSI 0.17** (persistence CSI 0.10) — decided on a ~P75 event score, not the median
- P10–P90 interval calibrated by split-conformal (CQR): 72% empirical coverage (80% target)
- Coupling shows in the data: `corr(ISI × aerosol, PM2.5) = +0.51`; ISI diurnal cycle 0.66 pre-dawn → 0.12 mid-afternoon
- Model: PM2.5 + PM10 + NO₂ LightGBM emulators → real CPCB AQI; 299-feature matrix; `models.train` / `models.backtest` (or `notebooks/train_colab.ipynb` on a T4)

## Architecture

```mermaid
flowchart LR
  cpcb[CPCB real-time<br/>data.gov.in] --> ing[ingest/]
  met[ERA5 + GFS<br/>Open-Meteo] --> ing
  firms[NASA FIRMS<br/>stubble fires] --> ing
  oaq[OpenAQ v3 + S3<br/>historical truth] --> ing
  ing --> feat[features/<br/>AQI · ISI · plume vector · aerosol→PBL]
  feat --> model[models/<br/>LightGBM P10/P50/P90 emulator]
  model --> apisvc[api/<br/>FastAPI + hourly refresh]
  apisvc --> web[web/<br/>React + MapLibre dashboard]
  wrf[wrfchem/<br/>offline validation] -. calibrates .-> feat
```

## Quick start (local)

```bash
python -m venv .venv && . .venv/Scripts/activate      # bin/activate on macOS/Linux
pip install -e ".[model,api,dev]"
cp .env.example .env                                  # keys optional (see below)

# one live cycle + train once (uses OpenAQ S3 archive, keyless)
python -m ingest.openaq --start 2025-10-01 --end 2026-02-15   # real CPCB history
python -m ingest.weather --history --start 2025-10-01 --end 2026-02-15
python -m ingest.firms   --history --start 2025-10-01 --end 2026-02-15
python -m features.build  --history --start 2025-10-01 --end 2026-02-15
python -m models.train --model lgbm
python -m models.backtest                             # prints the table above

# serve
uvicorn api.main:app --port 8000        # http://localhost:8000/docs
cd web && npm install && npm run dev    # http://localhost:5173
```

Or everything at once:

```bash
docker compose up --build
# dashboard  http://localhost:8080     API  http://localhost:8000/docs
```

**Live data only.** There is no bundled snapshot and no heuristic fallback forecast:
on boot the API ingests real CPCB + GFS + FIRMS data and runs the trained model, and
serves `503` on `/api/forecast` and friends until that first cycle completes (a few
seconds to a minute). `/api/stations`, `/api/health` and the station registry are up
immediately. It then refreshes hourly. The one non-live source is the trained model
file shipped in `models/registry/` — retrain it with `python -m models.train`.

### Optional: Postgres mirror

By default the API holds the live state in memory only. Set `DATABASE_URL` to also
mirror every refresh — observations, the 72-hour forecast, active alerts, a
per-station rollup and a `refresh_log` — into Postgres, where it persists across
restarts and is queryable from psql / Grafana / BI. Reads for the map and forecast
still come from the in-memory cache; `GET /api/history/{id}` is served from Postgres
when it is enabled. A missing or unreachable database is a silent no-op — the API
runs unchanged.

```bash
pip install -e ".[api,postgres]"           # adds SQLAlchemy + psycopg
export DATABASE_URL=postgresql+psycopg://vayucast:vayucast@localhost:5432/vayucast

# with Docker Compose — brings up an extra postgres:18 service:
echo "DATABASE_URL=postgresql+psycopg://vayucast:vayucast@db:5432/vayucast" >> .env
docker compose --profile db up --build
```

`GET /api/health` reports `"postgres_mirror": true` once it is wired up.

### Data source keys (all optional, all free)

| Feed | Register at | Env var |
| --- | --- | --- |
| CPCB real-time AQI | https://www.data.gov.in/ | `DATA_GOV_IN_API_KEY` |
| NASA FIRMS fire hotspots | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | `FIRMS_MAP_KEY` |
| OpenAQ historical ground truth | https://explore.openaq.org/ | `OPENAQ_API_KEY` (S3 archive needs no key) |
| ERA5 + GFS meteorology | Open-Meteo — no key | — |

## Repository

| Path | Contents |
| --- | --- |
| `config/` | 65-station NCR registry (rebuilt from the live feed), geo + physical constants |
| `ingest/` | CPCB, ERA5/GFS (Open-Meteo), NASA FIRMS, OpenAQ v3 + S3; `run_ingest` orchestrator |
| `aqi/` | Indian National AQI (CPCB method), vectorized |
| `features/` | Inversion Strength Index, stubble-plume transport vector, aerosol→PBL feedback, calendar/solar, builder |
| `models/` | LightGBM P10/P50/P90 emulator, walk-forward backtest, grouped-SHAP drivers, serving path |
| `wrfchem/` | Offline WRF-Chem namelists + FINN/EDGAR/mozbc runbook + `validate.py` |
| `api/` | FastAPI service + APScheduler hourly refresh (live data only, no snapshot) + optional Postgres mirror |
| `web/` | React + Vite + MapLibre dashboard (see `web/README.md`) |
| `.planning/` | gsd-core spec-driven workflow artifacts (PROJECT / REQUIREMENTS / ROADMAP / STATE) |

## Demo script

1. **Map** — NCR AQI choropleth + 65 station markers by CPCB category; fire hotspots + plume-transport line.
2. **Time slider** — drag *Now → +72 h*; the choropleth and the selected station's panel update.
3. **Station panel** — 72 h P50 line with P10–P90 uncertainty band, current vs peak AQI, health advisory.
4. **Why this forecast** — Inversion Strength Index + incoming-stubble-load meters, plume bearing, and SHAP contributions grouped as *inversion trapping / stubble transport / local emissions / wind ventilation / meteorology / time & season*.
5. **Alerts** — soonest "Very Poor" / "Severe" crossing across NCR, with lead time.
6. **Methodology** — model card + the live walk-forward backtest against real CPCB data. (The offline WRF-Chem validation figure appears here once that run is done — see `wrfchem/`.)

## License

MIT

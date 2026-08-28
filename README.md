# VayuCast

**Air Pollution–Weather Coupled Forecasting System for Delhi NCR**
Smart India Hackathon 2026 · Problem Statement **26082** · Ministry of Earth Sciences → NCMRWF

VayuCast produces a **72-hour, hourly, probabilistic AQI forecast** for the Delhi National
Capital Region that explicitly models the two-way meteorology–chemistry feedback standard
forecasts ignore:

- **Inversion → aerosol**: temperature inversions and a shallow planetary boundary layer trap
  PM2.5 near the surface.
- **Aerosol → inversion**: dense aerosol loading blocks sunlight, cooling the surface and
  further suppressing boundary-layer growth.

The live forecast is served by a fast ML emulator trained on engineered coupling features
(an **Inversion Strength Index** and a **stubble-plume transport vector**). A one-off,
offline **WRF-Chem** run validates the approach against a historical stubble-burning spike.

## Architecture

```
CPCB AQI ─┐
ERA5 / GFS ├─► ingest ─► features (AQI, ISI, stubble vector, aerosol→PBL) ─► ML emulator ─► FastAPI ─► React + MapLibre dashboard
NASA FIRMS ┘                                                                 (LightGBM → TFT)   (hourly refresh)
                                                              WRF-Chem (offline validation only)
```

See `.planning/ROADMAP.md` for the phase plan and `.planning/PROJECT.md` for scope and decisions.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on macOS/Linux
pip install -e ".[model,api,dev]"
cp .env.example .env                                # fill in keys when you have them (all optional)
pytest
```

### Data source keys (all optional for a first run)

| Feed | Where to register | Env var |
| --- | --- | --- |
| CPCB real-time AQI | https://www.data.gov.in/ | `DATA_GOV_IN_API_KEY` |
| NASA FIRMS fire hotspots | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | `FIRMS_MAP_KEY` |
| ERA5 history + GFS forecast | Open-Meteo (no key needed) | — |
| ERA5 high-fidelity (optional) | https://cds.climate.copernicus.eu/ | `CDSAPI_KEY` (extras: `gridded`) |

Without keys, ingestion falls back to cached snapshots so the demo still runs.

## Repository layout

| Path | Contents |
| --- | --- |
| `config/` | `stations.yaml` (NCR station registry), `settings.py` (paths, geo + physical constants) |
| `ingest/` | CPCB, ERA5, GFS, FIRMS ingestion → Parquet/DuckDB |
| `aqi/` | Indian National AQI (CPCB method) |
| `features/` | Inversion Strength Index, stubble-plume vector, aerosol→PBL feedback, feature builder |
| `models/` | LightGBM baseline, Temporal Fusion Transformer, training + backtest |
| `wrfchem/` | Offline WRF-Chem namelists, pipeline runbook, validation notebook |
| `api/` | FastAPI service + APScheduler hourly refresh |
| `web/` | React + Vite + MapLibre dashboard |
| `notebooks/` | EDA, feature checks, model evaluation |

## Status

Phase 1 (scaffold) in progress. Tracked in `.planning/STATE.md`.

## License

MIT

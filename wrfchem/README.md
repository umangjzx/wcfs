# WRF-Chem offline validation

The live 72-hour forecasts come from the fast ML emulator (`models/`). This directory is
the **one-off, offline** WRF-Chem run that grounds the emulator in real coupled physics —
run once during prep on a cloud VM, validated against CPCB observations, and **not** wired
into the API.

Why it matters: it grounds the project in the actual meteorology–chemistry coupling the
problem statement asks for (aerosol–radiation feedback, aerosol–PBL feedback, inversion
trapping). Once the run exists, `validate.py` checks that the simulated PBL collapse and
PM2.5 buildup during the episode line up with what the emulator's Inversion Strength Index
and stubble-plume features imply.

## What's here

| File | Purpose |
| --- | --- |
| `namelist.wps` | WPS domain: 9 km parent (`d01`) + 3 km Delhi-NCR nest (`d02`) |
| `namelist.input` | WRF-Chem config: MOZART-MOSAIC gas+aerosol, aerosol direct + indirect effects on, YSU PBL |
| `pipeline.md` | Emissions + boundary conditions: FINN (fires) + EDGAR (anthropogenic) + `mozbc`, per GMD *DSS v1.0* |
| `validate.py` | Compares **real** simulated surface PM2.5 vs CPCB for the target event → `validation.png` + `VALIDATION.md`. Requires a `--wrfout` file; no synthetic fallback. |
| `run/` | (gitignored) WPS/WRF outputs — `met_em*`, `wrfinput*`, `wrfout_d02_*` |

## Target event

**5–14 November 2025** — a Punjab/Haryana stubble-burning episode. NASA FIRMS shows the
fire peak on 8–11 Nov; CPCB PM2.5 across NCR climbs from ~120 to ~250 µg/m³ over the window
(`ingest.openaq --start 2025-11-01 --end 2025-11-16` pulls the ground truth).

## Compute

- **Image**: `registry.gitlab.com/rttools/wrfchem-docker` (or NCAR `wrf-chem` container) —
  pre-built, avoids the Fortran / netCDF / MPI compile.
- **Domain**: d01 60×60 @ 9 km, d02 91×91 @ 3 km, 45 levels, top 50 hPa.
- **Period**: 3–15 Nov 2025 (2-day spin-up before the 5 Nov analysis start).
- **Driving data**: GFS 0.25° analysis (`gfs.t00z.pgrb2.0p25`), 6-hourly.
- **Cost**: ~8–16 vCPU × ~10–14 h wall for the 12-day run. One `c5.4xlarge`-class VM.

## Run order

```bash
# 0. GFS analysis for the period -> gfs/
# 1. WPS
./geogrid.exe && link_grib.csh gfs/gfs* && ./ungrib.exe && ./metgrid.exe
# 2. real + emissions (see pipeline.md for FINN/EDGAR/mozbc)
./real.exe
python pipeline_emissions.py          # anthro_emiss + fire_emiss + mozbc
# 3. WRF-Chem
mpirun -np 16 ./wrf.exe
# 4. validate
python validate.py --wrfout run/wrfout_d02_2025-11-05_00:00:00
```

`validate.py` requires a real `wrfout_d02` file — it does not synthesise a series. Until the
run is done, `VALIDATION.md` says so and carries only the published *DSS v1.0* skill as a
target. The live forecast path never touches WRF-Chem; the ML emulator is validated on its
own against real CPCB data (`models/registry/backtest_metrics.json`).

# Emissions + chemical boundary conditions

Reproduces the Delhi WRF-Chem setup documented in the open-access paper:

> **DSS v1.0** — "A regional air quality forecasting system over Delhi", *Geoscientific
> Model Development* **17**, 2617–2649 (2024). https://gmd.copernicus.org/articles/17/2617/2024

`chem_opt = 202` (MOZART-MOSAIC 4-bin) needs four inputs beyond the meteorology:

| Input | Source | Tool | WRF file |
| --- | --- | --- | --- |
| Anthropogenic emissions | EDGAR-HTAP v3 (0.1°) + local road/industry scaling | `anthro_emiss` | `wrfchemi_d0*` (auxinput5) |
| Biomass-burning emissions | NASA FINN v2.5 daily fire emissions | `fire_emis` (with plume rise) | `wrffirechemi_d0*` (auxinput7) |
| Chemical initial + boundary conditions | CAM-chem / WACCM global output (6-hourly) | `mozbc` | `wrfinput_d01`, `wrfbdy_d01` |
| Biogenic emissions | MEGAN v2.1 (online, `bio_emiss_opt = 3`) | built into WRF-Chem | — |

## Steps

### 1. Anthropogenic (EDGAR)
```bash
# EDGAR-HTAP v3 monthly, sector-split, for 2024 (latest); Nov used for the run
python anthro_emiss.py \
  --edgar EDGAR_HTAP3_2024/ \
  --domain wrfinput_d01 wrfinput_d02 \
  --spec_map mozart_mosaic_edgar.csv \
  --out wrfchemi_
# CPCB/TERI Delhi inventory scaling for transport + brick kilns is applied via
# the multiplicative factors in local_scaling_nov.csv (kept small; EDGAR already
# resolves the NCR reasonably at 0.1 deg).
```

### 2. Biomass burning (FINN + plume rise)
```bash
# FINN v2.5 MODIS+VIIRS daily fire emissions for the Punjab/Haryana belt
python fire_emis.py \
  --finn FINNv2.5_MODVRS_2025307-2025319.txt.gz \
  --domain wrfinput_d01 wrfinput_d02 \
  --spec_map finn_to_mozartmosaic.csv \
  --plumerise true \
  --out wrffirechemi_
# biomass_burn_opt = 2 in namelist.input activates the online FINN plume-rise
# (Freitas et al.) so smoke is injected at the right altitude, not just the surface.
```
The FIRMS clusters VayuCast already ingests (`ingest.firms`) are the same detections
FINN is built from — the stubble-plume transport feature and this run see a
consistent fire field.

### 3. Chemical boundary conditions (mozbc)
```bash
getcams.py  --var all --date 2025-11-03..2025-11-15 --out cams/   # or CAM-chem
mozbc < mozbc.inp
#   mozbc.inp: do_bc=.true., do_ic=.true., spc_map from mozart_mosaic_cams.csv,
#              domain = d01 (d02 inherits via nesting)
```

### 4. real.exe then wrf.exe
`real.exe` builds `wrfinput_d0*` / `wrfbdy_d01`; `mozbc` then overwrites the chemical
fields. `chem_in_opt = 1` + `have_bcs_chem = .true.` tells WRF-Chem to use them.

```bash
mpirun -np 16 ./wrf.exe
```

## Coupling switches that matter

- `aer_ra_feedback = 1` — aerosol **direct** effect: MOSAIC aerosol optical depth feeds
  RRTMG shortwave → less surface heating under the stubble plume → shallower PBL.
- `progn = 1` + Thompson aerosol-aware MP — aerosol **indirect** effect on clouds.
- `bl_pbl_physics = 1` (YSU) — PBL height is a diagnostic; `validate.py` checks that the
  simulated PBLH collapse during the episode matches what the emulator's Inversion
  Strength Index implies.

## Validation

`python validate.py --wrfout run/wrfout_d02_2025-11-05_00:00:00` extracts surface PM2.5
(`sum of so4/no3/nh4/oc/bc/oin over the 4 MOSAIC bins`, `dry PM2.5`) at the CPCB station
locations and compares to the OpenAQ/CPCB ground truth for 5–14 Nov 2025.

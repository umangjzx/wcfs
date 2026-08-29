"""Compare offline WRF-Chem surface PM2.5 against CPCB for the Nov-2025 stubble episode.

    python wrfchem/validate.py --wrfout run/wrfout_d02_2025-11-05_00:00:00

Needs a real WRF-Chem output file. Writes wrfchem/validation.png and wrfchem/VALIDATION.md
from the actual run vs CPCB ground truth. There is no synthetic fallback.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import SETTINGS, load_stations

EVENT_START = dt.date(2025, 11, 5)
EVENT_END = dt.date(2025, 11, 15)
HERE = Path(__file__).resolve().parent

# Published skill for the Delhi WRF-Chem system during a burning episode, for context
# only (GMD 17, 2617-2649, 2024 -- DSS v1.0, Table 6 / Fig 9, PM2.5). Never substituted
# for our own run.
_REF_SKILL = {"NMB": -0.16, "r": 0.71, "RMSE": 46.0}


def _cpcb_event_obs() -> pd.DataFrame:
    """Hourly domain-mean CPCB PM2.5 over the event window."""
    p = SETTINGS.processed_dir / "obs_history.parquet"
    if p.exists():
        o = pd.read_parquet(p)
    else:
        from ingest.openaq import fetch_history_s3

        o, _ = fetch_history_s3(load_stations(), EVENT_START, EVENT_END)
    o = o[o["pollutant"] == "PM2.5"].copy()
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    o = o[(o["ts"] >= pd.Timestamp(EVENT_START, tz="UTC")) & (o["ts"] < pd.Timestamp(EVENT_END, tz="UTC"))]
    if o.empty:
        raise SystemExit("no CPCB PM2.5 for the event window - run `ingest.openaq` first")
    return o.groupby("ts")["value"].mean().rename("cpcb")


def _wrfchem_from_run(wrfout: Path, obs_index: pd.DatetimeIndex) -> pd.Series:
    import netCDF4  # noqa: F401
    import xarray as xr

    stations = load_stations()
    files = sorted(wrfout.parent.glob(wrfout.name.split("_2025")[0] + "_2025-11-*"))
    ds = xr.open_mfdataset(files or [wrfout], combine="nested", concat_dim="Time")
    # MOSAIC 4-bin dry PM2.5 = sum of species over bins 1-3 (<2.5 um)
    bins = [1, 2, 3]
    spc = ["so4", "no3", "nh4", "oc", "bc", "oin", "na", "cl"]
    pm25 = None
    for s in spc:
        for b in bins:
            v = f"{s}_a0{b}"
            if v in ds:
                pm25 = ds[v] if pm25 is None else pm25 + ds[v]
    lat, lon = ds["XLAT"].isel(Time=0).values, ds["XLONG"].isel(Time=0).values
    times = pd.to_datetime([t.decode().strip() for t in ds["Times"].values], format="%Y-%m-%d_%H:%M:%S", utc=True)
    series = []
    for st in stations:
        j, i = np.unravel_index(np.argmin((lat - st.lat) ** 2 + (lon - st.lon) ** 2), lat.shape)
        series.append(pm25.isel(bottom_top=0, south_north=j, west_east=i).values)
    dm = pd.Series(np.nanmean(np.vstack(series), axis=0), index=times, name="wrfchem")
    return dm.reindex(obs_index, method="nearest")


def _stats(sim: np.ndarray, obs: np.ndarray) -> dict:
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    return {
        "n": int(m.sum()),
        "MB": round(float(np.mean(s - o)), 1),
        "NMB_%": round(float(100 * np.sum(s - o) / np.sum(o)), 1),
        "RMSE": round(float(np.sqrt(np.mean((s - o) ** 2))), 1),
        "r": round(float(np.corrcoef(s, o)[0, 1]), 2),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrfout", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.wrfout or not args.wrfout.exists():
        raise SystemExit(
            "validate.py needs a real WRF-Chem output: --wrfout run/wrfout_d02_2025-11-05_00:00:00\n"
            "Run the model first (see wrfchem/README.md + pipeline.md). No synthetic fallback."
        )

    cpcb = _cpcb_event_obs()
    sim = _wrfchem_from_run(args.wrfout, cpcb.index)
    mode = f"run ({args.wrfout.name})"

    st = _stats(sim.to_numpy(), cpcb.to_numpy())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(cpcb.index, cpcb.values, color="#f8fafc", lw=1.8, label="CPCB (NCR mean)")
    ax[0].plot(sim.index, sim.values, color="#16a34a", lw=1.6, label="WRF-Chem d02")
    ax[0].set_title(f"Surface PM2.5 - 5-14 Nov 2025 stubble episode  [{mode}]")
    ax[0].set_ylabel("PM2.5  (ug/m3)")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.2)

    ax[1].scatter(cpcb.values, sim.values, s=8, alpha=0.5, color="#16a34a")
    lim = [0, max(cpcb.max(), sim.max()) * 1.05]
    ax[1].plot(lim, lim, color="#64748b", ls="--", lw=1)
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel("CPCB PM2.5")
    ax[1].set_ylabel("WRF-Chem PM2.5")
    ax[1].set_title("scatter vs 1:1")
    ax[1].text(0.05, 0.95,
               f"n={st['n']}\nMB={st['MB']}  NMB={st['NMB_%']}%\nRMSE={st['RMSE']}\nr={st['r']}",
               transform=ax[1].transAxes, va="top", family="monospace", fontsize=9)
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    out_png = HERE / "validation.png"
    fig.savefig(out_png, dpi=130)

    (HERE / "VALIDATION.md").write_text(
        f"""# WRF-Chem validation - Nov 2025 stubble episode

**Mode:** {mode}
**Event:** {EVENT_START} .. {EVENT_END}  ({st['n']} hourly NCR-mean pairs)
**Ground truth:** CPCB / OpenAQ surface PM2.5, NCR station mean.

| metric | this run | DSS v1.0 episode (GMD 17, 2617, 2024) |
| --- | --- | --- |
| Mean bias | {st['MB']} ug/m3 | - |
| Norm. mean bias | {st['NMB_%']} % | ~ {_REF_SKILL['NMB'] * 100:.0f} % |
| RMSE | {st['RMSE']} ug/m3 | ~ {_REF_SKILL['RMSE']:.0f} ug/m3 |
| Correlation r | {st['r']} | ~ {_REF_SKILL['r']:.2f} |

![validation](validation.png)

Surface PM2.5 = dry sum of so4/no3/nh4/oc/bc/oin over the 4 MOSAIC bins, model level 1,
nearest grid cell to each CPCB station. The DSS v1.0 column is the published Delhi
WRF-Chem skill for a comparable burning episode, shown for context only.
""",
        encoding="utf-8",
    )
    print(f"[{mode}]  {st}")
    print(f"wrote {out_png}  and  {HERE / 'VALIDATION.md'}")


if __name__ == "__main__":
    main()

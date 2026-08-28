"""Offline tests for weather (uv decomposition, block parsing) and FIRMS (parse, cluster)."""

import numpy as np

from ingest.firms import cluster_daily, parse_firms_csv
from ingest.weather import _parse_block, _uv


def test_uv_meteorological_convention():
    # wind FROM the north (0 deg) -> blowing toward south -> v negative, u ~ 0
    u, v = _uv(np.array([10.0]), np.array([0.0]))
    assert abs(u[0]) < 1e-9 and v[0] == -10.0
    # wind FROM the west (270) -> blowing toward east -> u positive
    u, v = _uv(np.array([5.0]), np.array([270.0]))
    assert u[0] == 5.0 and abs(v[0]) < 1e-9


def test_parse_block_maps_fields_and_lead_hours():
    import pandas as pd

    issue = pd.Timestamp("2024-11-05T00:00:00Z")
    block = {
        "hourly": {
            "time": ["2024-11-05T00:00", "2024-11-05T01:00", "2024-11-05T02:00"],
            "temperature_2m": [18.0, 17.5, 17.0],
            "boundary_layer_height": [120.0, 90.0, 80.0],
            "wind_speed_10m": [3.0, 3.0, 3.0],
            "wind_direction_10m": [0.0, 90.0, 180.0],
            "relative_humidity_2m": [80, 82, 85],
        }
    }
    df = _parse_block(block, "DEL-ito", "forecast", "openmeteo:gfs", issue)
    assert list(df["lead_h"]) == [0, 1, 2]
    assert df["t2m"].tolist() == [18.0, 17.5, 17.0]
    assert df["blh"].tolist() == [120.0, 90.0, 80.0]
    # wind from east (90) -> u negative
    assert df.iloc[1]["wind_u10"] < 0
    assert str(df["ts"].dt.tz) == "UTC"


def test_parse_block_empty():
    df = _parse_block({"hourly": {"time": []}}, "X", "reanalysis", "s", None)
    assert df.empty


FIRMS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
30.10,75.50,330.1,0.5,0.5,2024-11-05,0700,N,VIIRS,n,2.0NRT,290.0,12.5,D
30.12,75.52,340.0,0.5,0.5,2024-11-05,0700,N,VIIRS,h,2.0NRT,295.0,25.0,D
30.55,76.10,320.0,0.5,0.5,2024-11-05,0700,N,VIIRS,l,2.0NRT,288.0,5.0,D
30.11,75.51,331.0,0.5,0.5,2024-11-06,0700,N,VIIRS,n,2.0NRT,289.0,8.0,D
"""


def test_parse_firms_csv_and_confidence_mapping():
    det = parse_firms_csv(FIRMS_CSV, "VIIRS_SNPP_NRT")
    assert len(det) == 4
    assert set(det["confidence"]) <= {20.0, 60.0, 90.0}
    assert det["frp"].sum() == 50.5


def test_cluster_daily_grids_and_sums_frp():
    det = parse_firms_csv(FIRMS_CSV, "VIIRS_SNPP_NRT")
    clus = cluster_daily(det)
    # 2024-11-05: (30.10,75.50) and (30.12,75.52) fall in the same 0.1deg cell; (30.55,76.10) separate
    nov5 = clus[clus["date"] == "2024-11-05"]
    assert len(nov5) == 2
    top = nov5.sort_values("frp_sum", ascending=False).iloc[0]
    assert top["frp_sum"] == 37.5 and top["count"] == 2
    assert clus["date"].nunique() == 2


def test_cluster_daily_empty():
    import pandas as pd

    assert cluster_daily(pd.DataFrame()).empty

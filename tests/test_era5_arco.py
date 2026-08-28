"""Offline tests for ARCO-ERA5 pure helpers (no network / no heavy deps needed)."""

import numpy as np
import pytest

from ingest.era5_arco import _magnus_rh, _pick


def test_magnus_rh_saturation_and_dry():
    # dewpoint == temperature -> 100% RH
    assert _magnus_rh(np.array([20.0]), np.array([20.0]))[0] == pytest.approx(100.0, abs=1e-6)
    # large dewpoint depression -> low RH
    assert _magnus_rh(np.array([30.0]), np.array([5.0]))[0] < 25.0
    # clipped to [0, 100]
    out = _magnus_rh(np.array([10.0]), np.array([25.0]))  # unphysical Td>T
    assert out[0] == 100.0


def test_pick_first_available_variable():
    class DS:
        data_vars = {"surface_solar_radiation_downwards": 1, "2m_temperature": 1}

    assert _pick(DS(), ["surface_net_solar_radiation", "surface_solar_radiation_downwards"]) == (
        "surface_solar_radiation_downwards"
    )
    assert _pick(DS(), ["nonexistent"]) is None

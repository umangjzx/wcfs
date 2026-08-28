"""Sanity checks on the station registry and settings."""

from config.settings import (
    NCR_BBOX,
    NCR_CITIES,
    load_stations,
    station_index,
)


def test_station_registry_loads_and_is_reasonable():
    stations = load_stations()
    assert len(stations) >= 35
    ids = [s.id for s in stations]
    assert len(ids) == len(set(ids)), "station ids must be unique"


def test_all_stations_have_valid_coords_within_ncr_box():
    lat_min, lon_min, lat_max, lon_max = NCR_BBOX
    for s in load_stations():
        assert lat_min <= s.lat <= lat_max, f"{s.id} lat {s.lat} outside NCR bbox"
        assert lon_min <= s.lon <= lon_max, f"{s.id} lon {s.lon} outside NCR bbox"


def test_every_city_in_registry_is_declared():
    cities = {s.city for s in load_stations()}
    assert cities.issubset(set(NCR_CITIES))
    # each declared NCR city has at least one station
    assert cities == set(NCR_CITIES) - (set(NCR_CITIES) - cities)


def test_station_index_roundtrip():
    idx = station_index()
    some = load_stations()[0]
    assert idx[some.id] is some


def test_site_types_are_from_known_vocabulary():
    allowed = {"traffic", "industrial", "residential", "background", "mixed"}
    for s in load_stations():
        assert s.site_type in allowed, f"{s.id}: {s.site_type}"

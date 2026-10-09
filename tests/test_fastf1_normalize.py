import numpy as np, pandas as pd
from ingestion.normalize_fastf1 import normalize_laps, normalize_race_control, normalize_weather
from storage.db import connect, upsert


def laps_fixture():
    td = lambda v: pd.to_timedelta(v, unit="s")
    return pd.DataFrame({
        "Driver": ["VER", "VER", "HAM", "XXX"], "LapNumber": [1.0, 2.0, 1.0, 1.0],
        "LapTime": td([np.nan, 95.5, 96.0, 97.0]), "Sector1Time": td([30.0, 31.0, 31.5, 32.0]),
        "Sector2Time": td([35.0, 34.0, 34.5, 35.0]), "Sector3Time": td([np.nan, 30.5, 30.0, 30.0]),
        "Compound": ["MEDIUM", "MEDIUM", "HARD", None], "TyreLife": [1.0, 2.0, 3.0, np.nan],
        "Stint": [1.0, 1.0, 1.0, 1.0], "PitInTime": td([np.nan, 1000.0, np.nan, np.nan]),
        "PitOutTime": td([100.0, np.nan, np.nan, np.nan]), "Position": [1.0, 1.0, 2.0, 3.0],
        "TrackStatus": ["1", "14", "1", "1"], "IsAccurate": [False, True, True, True]})


CMAP = {"VER": "max_verstappen", "HAM": "hamilton"}


def test_normalize_laps_units_mapping_and_flags():
    out, unmatched = normalize_laps(laps_fixture(), "2023_01", CMAP)
    assert unmatched == {"XXX"} and len(out) == 3
    v2 = out[(out.driver_id == "max_verstappen") & (out.lap_number == 2)].iloc[0]
    assert v2.lap_time_s == 95.5 and v2.pit_in and not v2.pit_out and v2.track_status == "14" and v2.tyre_life == 2
    v1 = out[(out.driver_id == "max_verstappen") & (out.lap_number == 1)].iloc[0]
    assert np.isnan(v1.lap_time_s) and v1.pit_out and not v1.is_accurate


def test_normalize_is_tolerant_to_missing_columns():
    out, _ = normalize_laps(laps_fixture()[["Driver", "LapNumber", "LapTime"]], "2023_01", CMAP)
    assert len(out) == 3 and out["compound"].isna().all() and out["is_accurate"].tolist() == [False, True, True]


def test_load_into_duckdb_is_idempotent():
    con = connect()
    out, _ = normalize_laps(laps_fixture(), "2023_01", CMAP)
    for _ in range(2):
        upsert(con, "clean.laps", out, ["race_id", "driver_id", "lap_number"])
    assert con.execute("select count(*) from clean.laps").fetchone()[0] == 3
    assert con.execute("select count(*) from clean.laps where tyre_life is null").fetchone()[0] == 0


def test_weather_and_race_control():
    w = pd.DataFrame({"Time": pd.to_timedelta([0, 60], unit="s"), "AirTemp": [25.0, 25.5], "TrackTemp": [40.0, 41.0],
                      "Humidity": [50.0, 51.0], "Pressure": [1010.0, 1010.0], "WindSpeed": [2.0, 3.0],
                      "Rainfall": [False, True]})
    cw = normalize_weather(w, "2023_01")
    assert cw["t_offset_s"].tolist() == [0.0, 60.0] and cw["rainfall"].tolist() == [False, True]
    rc = pd.DataFrame({"Lap": [1.0, 12.0], "Category": ["Flag", "SafetyCar"], "Message": ["GREEN", "SAFETY CAR DEPLOYED"]})
    crc = normalize_race_control(rc, "2023_01")
    assert crc["lap"].tolist() == [1, 12] and crc["category"].tolist() == ["Flag", "SafetyCar"]
    con = connect()
    con.register("_w", cw); con.execute("insert into clean.weather by name select * from _w")
    con.register("_r", crc); con.execute("insert into clean.race_control by name select * from _r")
    assert con.execute("select count(*) from clean.weather").fetchone()[0] == 2



def test_text_nulls_stored_as_strings_become_real_nulls():
    """Regresión: el RAW guardado como texto trae 'nan'/'None'; deben ser nulos reales en clean."""
    lf = laps_fixture()
    lf["Compound"] = ["MEDIUM", "nan", "None", "SOFT"]
    lf["TrackStatus"] = ["1", "nan", "None", "1"]
    out, _ = normalize_laps(lf, "2023_01", {"VER": "v", "HAM": "h", "XXX": "x"})
    assert out["compound"].isna().sum() == 2 and out["track_status"].isna().sum() == 2
    assert set(out["compound"].dropna()) == {"MEDIUM", "SOFT"}
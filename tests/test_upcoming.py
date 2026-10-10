import numpy as np, pandas as pd, pytest
from features.build import build_features
from features.upcoming import NoQualifying, upcoming_rows
from ingestion.jolpica import merge_race_pages
from models.predict import predict_race
from tests.test_features import raw_toy


def _page(race_extra):
    race = {"season": "2026", "round": "17", "raceName": "Singapore Grand Prix", "date": "2026-10-11",
            "Circuit": {"circuitId": "marina_bay", "circuitName": "Marina Bay Street Circuit",
                        "Location": {"lat": "1.2914", "long": "103.864", "locality": "Marina Bay", "country": "Singapore"}},
            **race_extra}
    return {"MRData": {"total": "1", "RaceTable": {"Races": [race]}}}


def _q(drv, team, pos, q1):
    return {"position": str(pos), "Driver": {"driverId": drv, "code": drv[:3].upper(), "givenName": "A", "familyName": drv},
            "Constructor": {"constructorId": team, "name": team}, "Q1": q1, "Q2": "", "Q3": ""}


def test_upcoming_rows_from_calendar_and_qualifying():
    info = merge_race_pages([_page({})])
    quali = merge_race_pages([_page({"QualifyingResults": [_q("a", "t1", 1, "1:30.100"), _q("b", "t2", 2, "1:30.500")]})],
                             "QualifyingResults")
    up = upcoming_rows(info, quali, {"b": 0})
    assert list(up["race_id"].unique()) == ["2026_17"] and up["circuit_id"].iloc[0] == "marina_bay"
    assert up["grid"].tolist() == [1.0, 0.0] and np.isclose(up["q1_s"].iloc[0], 90.1)
    assert up["finish_position"].isna().all()


def test_no_qualifying_yet_is_reported():
    with pytest.raises(NoQualifying):
        upcoming_rows(merge_race_pages([_page({})]), merge_race_pages([_page({})], "QualifyingResults"))


def test_prediction_is_identical_whether_or_not_the_race_results_exist():
    """Fuga: predecir una carrera ocultando sus resultados da exactamente lo mismo que con ellos presentes."""
    raw = raw_toy(seasons=range(2018, 2023), races=10)
    full = predict_race(build_features(raw), "2022_10")
    hidden = raw.copy()
    m = hidden["race_id"] == "2022_10"
    hidden.loc[m, "finish_position"], hidden.loc[m, "points"] = np.nan, 0.0
    hidden.loc[m, "classified"], hidden.loc[m, "race_pace_pct"] = True, np.nan
    pd.testing.assert_frame_equal(full, predict_race(build_features(hidden), "2022_10"))



def test_upcoming_rows_from_a_handwritten_table():
    from features.upcoming import upcoming_rows_from_table
    q = pd.DataFrame({"driver_id": ["a", "b", "c"], "position": [1, 2, np.nan], "q1": ["1:30.100", None, "1:31.000"]})
    up = upcoming_rows_from_table(2026, 17, "marina_bay", q, {"a": "t1", "b": "t2", "c": "t3"}, {"b": 0})
    assert up["driver_id"].tolist() == ["a", "b"] and up["team_id"].tolist() == ["t1", "t2"]   # c sin posición: no corre
    assert up["grid"].tolist() == [1.0, 0.0] and np.isclose(up["q1_s"].iloc[0], 90.1) and np.isnan(up["q1_s"].iloc[1])
    assert up["race_id"].iloc[0] == "2026_17" and up["circuit_id"].iloc[0] == "marina_bay"
    with pytest.raises(ValueError):
        upcoming_rows_from_table(2026, 17, "marina_bay", q, {"a": "t1"})            # piloto sin historial
    with pytest.raises(NoQualifying):
        upcoming_rows_from_table(2026, 17, "marina_bay", q.assign(position=np.nan), {"a": "t1", "b": "t2", "c": "t3"})
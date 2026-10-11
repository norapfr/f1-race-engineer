import numpy as np, pandas as pd
from pipelines.quali_from_fastf1 import best_times, classification_from_segments


def laps(rows):
    return pd.DataFrame({"Driver": [r[0] for r in rows],
                         "LapTime": pd.to_timedelta(np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=float), unit="s"),
                         "Deleted": [r[2] for r in rows]})


def test_deleted_and_missing_laps_are_ignored():
    b = best_times(laps([("E", 89.0, True), ("E", 93.0, False), ("D", None, False)]))
    assert b["E"] == 93.0 and "D" not in b.index


def test_classification_order_by_last_segment_reached():
    q1 = laps([("A", 90.0, False), ("B", 91.0, False), ("C", 92.0, False), ("D", None, False), ("E", 89.0, True),
               ("E", 93.0, False)])
    q2 = laps([("A", 89.5, False), ("B", 90.5, False), ("C", 91.5, False)])
    q3 = laps([("A", 89.0, False), ("B", 88.9, False)])
    t = classification_from_segments(q1, q2, q3, ["A", "B", "C", "D", "E", "F"])
    assert t["code"].tolist() == ["B", "A", "C", "E", "D", "F"]          # F no salió: último
    assert t["position"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.isclose(t.set_index("code").loc["E", "q1"], 93.0) and np.isnan(t.set_index("code").loc["D", "q1"])
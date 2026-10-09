import numpy as np, pandas as pd
from features.pace import race_pace_table


def laps(compound="MEDIUM"):
    rows = []
    offsets = {"a": -1.0, "b": -0.5, "c": 0.0, "d": 0.5, "e": 1.0, "f": 2.0}   # % respecto a la referencia
    for drv, off in offsets.items():
        for lap in range(1, 31):
            t = 90.0 * (1 + off / 100) - 0.03 * lap           # combustible: todos mejoran igual
            row = dict(race_id="2023_01", driver_id=drv, lap_number=lap, lap_time_s=t, compound=compound,
                       is_accurate=True, pit_in=False, pit_out=False, track_status="1")
            if lap == 1:
                row["lap_time_s"] = t + 8                       # salida lanzada: se descarta
            if lap == 10:
                row["track_status"] = "4"; row["lap_time_s"] = t + 25   # safety car: se descarta
            if lap in (15, 16) and drv == "c":
                row["pit_in" if lap == 15 else "pit_out"] = True; row["lap_time_s"] = t + 20
            if lap == 20 and drv == "e":
                row["lap_time_s"] = t + 40                      # vuelta absurda: fuera por outlier
            rows.append(row)
    return pd.DataFrame(rows)


def test_pace_orders_drivers_and_ignores_unclean_laps():
    out = race_pace_table(laps()).set_index("driver_id")
    assert list(out["race_pace_pct"].sort_values().index) == ["a", "b", "c", "d", "e", "f"]
    # referencia = mediana del campo (0.25% sobre el offset 0): a -1.0 -> -1.25 aprox., f +2.0 -> +1.75 aprox.
    assert np.isclose(out.loc["a", "race_pace_pct"], -1.25, atol=0.05) and np.isclose(out.loc["f", "race_pace_pct"], 1.75, atol=0.05)
    assert out["n_clean_laps"].max() <= 28                         # sin vuelta 1 ni safety car


def test_wet_race_and_too_few_clean_laps_are_nan():
    assert race_pace_table(laps("INTERMEDIATE"))["race_pace_pct"].isna().all()
    d = laps()
    d.loc[(d.driver_id == "b") & (d.lap_number > 6), "track_status"] = "4"   # solo 4 vueltas limpias
    out = race_pace_table(d).set_index("driver_id")
    assert np.isnan(out.loc["b", "race_pace_pct"]) and not np.isnan(out.loc["a", "race_pace_pct"])
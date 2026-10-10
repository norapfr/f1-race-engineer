import numpy as np, pandas as pd
from features.degradation import clean_dry_laps, stint_slopes

DEG, FUEL = 0.05, 0.03    # s/vuelta de degradación y de combustible (el coche se aligera)


def laps():
    rows = []
    for i, p in enumerate(range(14, 30, 2)):          # 8 pilotos con la parada en vueltas distintas
        for lap in range(1, 51):
            stint, age = (1, lap) if lap <= p else (2, lap - p)
            rows.append(dict(race_id="2023_01", driver_id=f"d{i}", lap_number=lap, stint=stint,
                             compound="MEDIUM" if stint == 1 else "HARD", tyre_life=age,
                             lap_time_s=90 - FUEL * lap + DEG * age, is_accurate=True, track_status="1",
                             pit_in=lap == p, pit_out=lap == p + 1))
    return pd.DataFrame(rows)


def test_raw_slope_is_degradation_minus_fuel_and_excludes_pit_laps():
    ok = clean_dry_laps(laps())
    assert not ((ok["lap_number"] == 1)).any()
    s = stint_slopes(ok)
    assert len(s) == 16 and np.allclose(s["slope_raw"], DEG - FUEL, atol=1e-9)


def test_relative_slope_underestimates_degradation_when_everyone_ages_together():
    s = stint_slopes(clean_dry_laps(laps()))
    assert (s["slope_rel"] < DEG - 0.01).all()     # la degradación común al campo desaparece de la medida relativa


def test_wet_race_is_excluded():
    d = laps()
    d.loc[d["lap_number"] < 10, "compound"] = "INTERMEDIATE"
    assert clean_dry_laps(d).empty
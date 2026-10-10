import numpy as np, pandas as pd
from models.tyre_degradation import COMPOUNDS, DegFit, curve_tables, demean, fit_degradation

TRUE_G = {"SOFT": lambda a: 0.08 * a + 0.05 * np.maximum(0, a - 10), "MEDIUM": lambda a: 0.04 * a + 0.02 * np.maximum(0, a - 20),
          "HARD": lambda a: 0.025 * a}
TRUE_OFF = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5}


def simulate(n_races=40, n_drv=16, seed=0, noise=0.25):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        base = 90 + rng.normal(0, 5)
        for d in range(n_drv):
            alpha = rng.normal(0, 0.4)
            first = int(rng.integers(10, 26))
            pits = [first] if rng.random() < 0.7 else [first, int(rng.integers(first + 8, 46))]
            comps = rng.choice(COMPOUNDS, len(pits) + 1)
            stint, age = 0, 0
            for lap in range(1, 56):
                age += 1
                c = comps[stint]
                t = base - 0.035 * lap + alpha + TRUE_OFF[c] + TRUE_G[c](age) + rng.normal(0, noise)
                rows.append(dict(race_id=f"r{r}", driver_id=f"d{d}", lap_number=lap, compound=c, tyre_life=age, lap_time_s=t))
                if stint < len(pits) and lap == pits[stint]:
                    stint, age = stint + 1, 0
    return pd.DataFrame(rows)


def test_demean_removes_group_means():
    rng = np.random.default_rng(1)
    g1, g2 = rng.integers(0, 20, 500), rng.integers(0, 15, 500)
    Z = demean(rng.normal(size=(500, 3)), [g1, g2])
    for g in (g1, g2):
        assert np.abs(pd.DataFrame(Z).groupby(g).mean().to_numpy()).max() < 1e-6


def test_recovers_known_degradation_curves_and_offsets_within_standard_errors():
    fit = fit_degradation(simulate())
    assert isinstance(fit, DegFit) and fit.n_races == 40
    for c in COMPOUNDS:
        for age in (10, 20, 30):
            est, se = fit.loss(c, age)
            truth = TRUE_G[c](age) - TRUE_G[c](1)
            assert se > 0 and abs(est - truth) < 3 * se + 0.03, (c, age, est, truth, se)
    for c in ("SOFT", "HARD"):
        est, se = fit.fresh_offset(c)
        truth = (TRUE_OFF[c] + TRUE_G[c](1)) - (TRUE_OFF["MEDIUM"] + TRUE_G["MEDIUM"](1))
        assert abs(est - truth) < 3 * se + 0.03, (c, est, truth, se)


def test_curve_tables_shape():
    table, fresh = curve_tables(fit_degradation(simulate(n_races=15)))
    assert list(table.columns) == list(COMPOUNDS) and len(table) == 5 and set(fresh) == set(COMPOUNDS)



def test_by_season_tables():
    from pipelines.degradation_by_season import by_season
    d = simulate(n_races=20)
    d["race_id"] = [f"{2022 + int(r[1:]) % 2}_{r[1:]}" for r in d["race_id"]]
    t = by_season(d)
    assert list(t["perdida_10"].index) == [2022, 2023] and list(t["perdida_10"].columns) == list(COMPOUNDS)
    assert t["carreras"].to_dict() == {2022: 10, 2023: 10}
import numpy as np, pandas as pd
from features.baselines import SlotBaseline, metrics, prepare
from pipelines.evaluate_baselines import evaluate


def toy(seasons=range(2018, 2023), races=20, n=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for r in range(1, races + 1):
            grid = rng.permutation(n) + 1
            fin = (grid + rng.normal(0, 1.5, n)).argsort().argsort() + 1
            for i in range(n):
                rows.append({"race_id": f"{s}_{r:02d}", "season": s, "round": r, "driver_id": f"d{i}", "team_id": "t",
                             "grid": int(grid[i]), "finish_position": int(fin[i]),
                             "points": {1: 25, 2: 18, 3: 15}.get(int(fin[i]), 0), "quali_pos": float(grid[i])})
    out = prepare(pd.DataFrame(rows))
    out["prior_driver_form_rank"] = out["grid"].astype(float)   # columna sintética para el baseline 4
    return out


def test_grid_baseline_beats_chance_and_win_probs_sum_to_one():
    df = toy()
    res = evaluate(df)
    pooled = res[res["season"] == "todas"].set_index("baseline")
    assert pooled.loc["1_grid", "brier_win"] < pooled.loc["0_azar", "brier_win"]
    assert pooled.loc["1_grid", "winner_acc"] > pooled.loc["0_azar", "winner_acc"]
    tr, te = df[df["season"] < 2022], df[df["season"] == 2022]
    pw, _, _ = SlotBaseline("grid_eff").fit(tr).predict(te)
    assert np.allclose(pd.Series(pw).groupby(te["race_id"].to_numpy()).sum(), 1.0)


def test_standings_slot_ignores_the_race_itself():
    """Alterar los puntos de la carrera R no puede cambiar su propio ranking de standings."""
    df = toy()
    R = "2020_07"
    base = df[df["race_id"] == R]["standings_slot"].to_numpy()
    pert = df.drop(columns=["standings_slot", "pts_before", "prev_total"]).copy()
    pert.loc[pert["race_id"] == R, "points"] = 999.0
    out = prepare(pert)
    assert np.array_equal(out[out["race_id"] == R]["standings_slot"].to_numpy(), base)


def test_pit_lane_grid_zero_is_last():
    df = toy()
    df.loc[0, "grid"] = 0
    out = prepare(df.drop(columns=["field", "grid_eff", "quali_slot", "pts_before", "prev_total", "standings_slot"]))
    assert out.loc[0, "grid_eff"] == out.loc[0, "field"]
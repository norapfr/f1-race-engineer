import numpy as np, pandas as pd, pytest
from features.availability import LeakageError, check_features
from features.build import FEATURES, TARGETS, build_features


def raw_toy(seasons=range(2019, 2023), races=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for r in range(1, races + 1):
            grid = rng.permutation(6) + 1
            fin = (grid + rng.normal(0, 1.5, 6)).argsort().argsort() + 1
            for i in range(6):
                rows.append({"race_id": f"{s}_{r:02d}", "season": s, "round": r, "circuit_id": f"c{r % 4}",
                             "driver_id": f"d{i}", "team_id": f"t{i // 2}", "grid": int(grid[i]),
                             "finish_position": int(fin[i]), "points": float({1: 25, 2: 18, 3: 15}.get(int(fin[i]), 0)),
                             "classified": bool(rng.random() > 0.1), "quali_pos": float(grid[i]),
                             "q1_s": 90 + 0.1 * grid[i] + rng.normal(0, 0.05),
                             "race_pace_pct": float(rng.normal(0, 1))})
    return pd.DataFrame(rows)


def test_all_features_are_pre_race_and_targets_are_not():
    check_features(FEATURES)
    with pytest.raises(LeakageError):
        check_features(FEATURES + ["target_win"])
    assert not set(FEATURES) & set(TARGETS)


def test_perturbing_race_results_does_not_change_that_races_features():
    """Cambiar resultado, puntos y abandonos de la carrera R no puede alterar NINGUNA de sus features."""
    raw = raw_toy()
    R = "2021_05"
    base = build_features(raw)
    p = raw.copy()
    m = p["race_id"] == R
    p.loc[m, "finish_position"] = np.random.default_rng(9).permutation(p.loc[m, "finish_position"].to_numpy())
    p.loc[m, "points"] = 999.0
    p.loc[m, "classified"] = ~p.loc[m, "classified"]
    p.loc[m, "race_pace_pct"] = 99.0
    pert = build_features(p)
    pd.testing.assert_frame_equal(base[base["race_id"] == R][FEATURES], pert[pert["race_id"] == R][FEATURES])
    nxt = "2021_06"   # sanity: la carrera siguiente SÍ ve el pasado alterado
    assert not base[base["race_id"] == nxt][FEATURES].equals(pert[pert["race_id"] == nxt][FEATURES])


def test_first_race_has_no_history_and_teammate_delta_is_symmetric():
    out = build_features(raw_toy())
    first = out[out["race_id"] == "2019_01"]
    assert first["prior_driver_finish_ewm"].isna().all() and (first["prior_driver_n_races"] == 0).all()
    t0 = out[(out["race_id"] == "2020_03") & (out["team_id"] == "t0")]["quali_teammate_delta"]
    assert np.isclose(t0.sum(), 0.0)


def test_evaluate_runs_on_the_real_feature_table():
    """Regresión: evaluate debe funcionar con la salida de build_features (no con columnas auxiliares)."""
    from pipelines.evaluate_baselines import evaluate
    res = evaluate(build_features(raw_toy()))
    assert set(res["baseline"]) == {"0_azar", "1_grid", "2_standings", "3_quali", "4_forma"}


def test_pace_features_exist_and_are_prior_only():
    out = build_features(raw_toy())
    assert out.loc[out["race_id"] == "2019_01", "prior_driver_pace_ewm"].isna().all()
    assert out["prior_team_pace_rank"].dropna().between(1, 3).all()
    assert out["prior_team_pace_gap"].dropna().ge(0).all()



def test_features_do_not_depend_on_input_row_order():
    """Reproducibilidad: el orden de filas que devuelva la base de datos no puede cambiar las features."""
    raw = raw_toy()
    shuffled = raw.sample(frac=1, random_state=0).reset_index(drop=True)
    pd.testing.assert_frame_equal(build_features(raw), build_features(shuffled))
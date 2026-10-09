import numpy as np, pandas as pd, pytest
from features.availability import Avail, LeakageError, check_features, classify
from features.temporal import assert_temporal_split, prior_ewm, prior_window_mean, walk_forward


def test_registry_rules():
    assert classify("grid") == Avail.POST_QUALI
    assert classify("prior_finish_mean_3") == Avail.PRE_WEEKEND
    assert classify("target_win") == Avail.POST_RACE


def test_post_race_feature_is_rejected():
    with pytest.raises(LeakageError):
        check_features(["grid", "prior_finish_mean_3", "points"])


def test_unregistered_column_fails_closed():
    with pytest.raises(LeakageError):
        check_features(["mystery_feature"])


def test_valid_feature_set_passes():
    check_features(["race_id", "driver_id", "grid", "quali_gap_to_pole", "prior_finish_mean_5", "circuit_length_km"])


def test_actual_weather_is_not_allowed_but_forecast_is():
    with pytest.raises(LeakageError):
        check_features(["actual_rainfall"])
    check_features(["forecast_rain_prob"])


def test_temporal_split():
    assert_temporal_split([201801, 201802], [201901])
    with pytest.raises(LeakageError):
        assert_temporal_split([201801, 201905], [201901])


def test_walk_forward_expanding():
    folds = list(walk_forward(list(range(2018, 2025)), min_train=4))
    assert folds[0] == {"train": [2018, 2019, 2020, 2021], "val": 2022, "test": 2023}
    assert folds[-1]["test"] == 2024
    assert all(max(f["train"]) < f["val"] < f["test"] for f in folds)


def _toy(n_races=10, n_drivers=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"season": 2023, "round": r + 1, "driver_id": f"d{d}", "finish_position": int(rng.integers(1, 21))}
            for r in range(n_races) for d in range(n_drivers)]
    return pd.DataFrame(rows)


def test_perturbation_target_race_result_does_not_change_its_own_features():
    """Test de leakage más fuerte: alterar el resultado de la carrera R no puede alterar sus features."""
    df = _toy()
    base = df.assign(f_mean=prior_window_mean(df, "driver_id", "finish_position", 3),
                     f_ewm=prior_ewm(df, "driver_id", "finish_position", 2.0))
    R = 7
    pert = df.copy()
    pert.loc[pert["round"] == R, "finish_position"] = 999
    pert = pert.assign(f_mean=prior_window_mean(pert, "driver_id", "finish_position", 3),
                       f_ewm=prior_ewm(pert, "driver_id", "finish_position", 2.0))
    a, b = base[base["round"] == R], pert[pert["round"] == R]
    pd.testing.assert_series_equal(a["f_mean"], b["f_mean"])
    pd.testing.assert_series_equal(a["f_ewm"], b["f_ewm"])
    # sanity: la carrera siguiente SÍ debe verse afectada (el pasado informa al futuro)
    assert not base[base["round"] == R + 1]["f_mean"].equals(pert[pert["round"] == R + 1]["f_mean"])


def test_first_race_has_no_history():
    df = _toy()
    f = prior_window_mean(df, "driver_id", "finish_position", 3)
    assert f[df["round"] == 1].isna().all()

import numpy as np, pandas as pd
from features.build import build_features
from models.predict import predict_race
from tests.test_features import raw_toy


def feats():
    return build_features(raw_toy(seasons=range(2018, 2023), races=10))


def test_predict_race_probabilities_are_coherent():
    out = predict_race(feats(), "2022_05")
    assert len(out) == 6
    assert np.isclose(out["p_win"].sum(), 1.0, atol=1e-2) and np.isclose(out["p_podium"].sum(), 3.0, atol=1e-2)
    assert out[["p_win", "p_podium", "p_top5", "p_top10"]].ge(0).all().all() and out["p_podium"].ge(out["p_win"] - 1e-9).all()


def test_prediction_ignores_the_race_itself_and_everything_after():
    """Fuga: cambiar los resultados de la carrera predicha y de las posteriores no puede cambiar la predicción."""
    df = feats()
    base = predict_race(df, "2022_05")
    pert = df.copy()
    m = (pert["season"] * 100 + pert["round"]) >= 202205
    rng = np.random.default_rng(5)
    for c in ["target_win", "target_podium", "target_top5", "target_top10", "target_finish", "target_points"]:
        pert.loc[m, c] = rng.permutation(pert.loc[m, c].to_numpy())
    pd.testing.assert_frame_equal(base, predict_race(pert, "2022_05"))
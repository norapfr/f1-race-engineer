import numpy as np, pandas as pd
from features.build import build_features
from models.predict import predict_race
from pipelines.score_prediction import score
from tests.test_features import raw_toy


def test_score_compares_models_on_the_same_drivers():
    df = build_features(raw_toy(seasons=range(2018, 2023), races=10))
    pred = predict_race(df, "2022_10")
    real = df.loc[df["race_id"] == "2022_10", ["driver_id", "target_finish"]].rename(columns={"target_finish": "finish_position"})
    uniform = pd.DataFrame({"driver_id": real["driver_id"], "p_win": 1 / 6, "p_podium": 0.5, "exp_finish": 3.5})
    t = score({"ens": pred[["driver_id", "p_win", "p_podium", "exp_finish"]], "azar": uniform}, real)
    assert list(t["modelo"]) == ["ens", "azar"] and (t["pilotos"] == 6).all()
    assert np.isclose(t.loc[1, "logloss_win"], -(np.log(1 / 6) + 5 * np.log(5 / 6)) / 6)
    assert 0 < t.loc[0, "P(ganador real)"] < 1 and t["MAE_posicion"].gt(0).all()
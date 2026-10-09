import numpy as np
from features.build import build_features
from models.race_model import FAMILIES, normalize_by_race, predict_season, recalibrate
from pipelines.train_eval import run
from tests.test_features import raw_toy


def feats():
    return build_features(raw_toy(seasons=range(2018, 2023), races=10))


def test_normalize_by_race_sums():
    ids = np.array(["a"] * 6 + ["b"] * 6)
    p = normalize_by_race(np.random.default_rng(0).random(12), ids, 3)
    assert np.allclose([p[:6].sum(), p[6:].sum()], 3.0, atol=1e-3) and p.max() <= 1


def test_changing_test_season_labels_does_not_change_predictions_or_recalibration():
    """Fuga: ni las predicciones ni su recalibración de la temporada t pueden depender de los resultados de t."""
    df = feats()
    t = 2022
    pert = df.copy()
    m = pert["season"] == t
    rng = np.random.default_rng(3)
    for c in ["target_win", "target_podium", "target_top5", "target_top10"]:
        pert.loc[m, c] = rng.permutation(pert.loc[m, c].to_numpy())
    pert.loc[m, "target_finish"] = rng.permutation(pert.loc[m, "target_finish"].to_numpy())
    for fam in FAMILIES:
        te_a, a, cfg_a = predict_season(fam, df, t)
        te_b, b, cfg_b = predict_season(fam, pert, t)
        assert cfg_a == cfg_b
        for k in a:
            assert np.allclose(a[k], b[k]), (fam.name, k)
        ca, cb = recalibrate(a, te_a), recalibrate(b, te_b)
        for k in ca:
            assert np.allclose(ca[k], cb[k]), (fam.name, "cal", k)
            assert np.allclose(pd_sum(ca[k], te_a), {"p_win": 1, "p_pod": 3}[k], atol=1e-3)


def pd_sum(p, te):
    import pandas as pd
    return pd.Series(p).groupby(te["race_id"].to_numpy()).sum().to_numpy()


def test_run_produces_all_tables():
    out = run(feats(), exclude_season=2022)
    names = {"1_grid", "logit", "lgbm", "logit_pace", "lgbm_pace", "ens"}
    assert names | {"3_quali"} <= set(out["metrics"]["modelo"])
    assert set(out["diffs"]["modelo"]) == names == set(out["diffs_sin"]["modelo"])
    assert set(out["ablation"]["modelo"]) == {"logit_pace vs logit", "lgbm_pace vs lgbm"}
    assert out["diffs"]["n_races"].iloc[0] == 20 and out["diffs_sin"]["n_races"].iloc[0] == 10
    assert out["best"] in ("logit", "lgbm", "logit_pace", "lgbm_pace", "ens") and len(out["calib_win"]) > 0
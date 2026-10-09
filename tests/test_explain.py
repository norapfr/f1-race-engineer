import numpy as np, pandas as pd
from features.build import FEATURES, build_features
from models.explain import explain_race, explain_table, gbm_contrib, logit_contrib
from models.race_model import FAMILIES
from tests.test_features import raw_toy


def feats():
    return build_features(raw_toy(seasons=range(2018, 2023), races=10))


def test_contributions_are_additive_for_both_models():
    """Contribuciones + valor base = puntuación del modelo (propiedad de SHAP y de los modelos lineales)."""
    df = feats()
    past, te = df[df["season"] < 2022], df[df["season"] == 2022].reset_index(drop=True)
    for fam in FAMILIES:
        clf = fam.make_clf(fam.candidates[0]).fit(past[FEATURES], past["target_win"])
        if fam.name == "logit":
            C, base = logit_contrib(clf, te[FEATURES])
            raw = clf.decision_function(te[FEATURES])
        else:
            C, base = gbm_contrib(clf, te[FEATURES])
            raw = np.mean([m.predict(te[FEATURES], raw_score=True) for m in clf.models], axis=0)
        assert np.allclose(C.sum(axis=1).to_numpy() + base, raw, atol=1e-6), fam.name


def test_explain_race_and_table():
    C = explain_race(feats(), "2022_05")
    assert C.shape == (6, len(FEATURES)) and C.index.is_unique
    t = explain_table(C, pd.Series({d: d.upper() for d in C.index}), list(C.index)[:2])
    assert list(t.columns) == ["piloto", "sube", "baja"] and len(t) == 2


def test_explanation_ignores_the_race_itself_and_everything_after():
    df = feats()
    base = explain_race(df, "2022_05")
    pert = df.copy()
    m = (pert["season"] * 100 + pert["round"]) >= 202205
    rng = np.random.default_rng(7)
    for c in ["target_win", "target_podium", "target_top5", "target_top10", "target_finish", "target_points"]:
        pert.loc[m, c] = rng.permutation(pert.loc[m, c].to_numpy())
    pd.testing.assert_frame_equal(base, explain_race(pert, "2022_05"))



def test_groups_partition_all_features_and_preserve_the_total():
    from models.explain import GROUPS, explain_groups, group_contrib
    flat = [f for cols in GROUPS.values() for f in cols]
    assert sorted(flat) == sorted(FEATURES)                       # cada feature en exactamente un grupo
    C = explain_race(feats(), "2022_05")
    assert np.allclose(group_contrib(C).sum(axis=1), C.sum(axis=1))
    M = explain_groups(C, pd.Series({d: d for d in C.index}), list(C.index)[:3])
    assert list(M.columns) == ["piloto"] + list(GROUPS) and (M.iloc[:, 1:] > 0).all().all()
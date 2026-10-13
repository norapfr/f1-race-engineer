import numpy as np, pandas as pd
from features.build import build_features
from simulation.evaluate import evaluate_simulator, simulate_season
from simulation.position_sim import (DnfModel, fit_pl, log_strength, position_matrix, simulate_positions, summarize)
from tests.test_features import raw_toy


def test_simulation_is_coherent_reproducible_and_matches_plackett_luce():
    s = np.log(np.array([5.0, 3.0, 1.0, 1.0]))
    pos = simulate_positions(s, np.zeros(4), n_sims=40_000, seed=1)
    P = position_matrix(pos)
    assert np.allclose(P.sum(axis=1), 1) and np.allclose(P.sum(axis=0), 1)            # cada piloto y cada plaza suman 1
    assert np.allclose(P[:, 0], np.exp(s) / np.exp(s).sum(), atol=0.01)               # P(ganar) = fuerza relativa
    assert np.array_equal(pos, simulate_positions(s, np.zeros(4), n_sims=40_000, seed=1))
    assert not np.array_equal(pos, simulate_positions(s, np.zeros(4), n_sims=40_000, seed=2))


def test_retirements_finish_behind_everyone_who_finishes():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    P = position_matrix(simulate_positions(np.array([9.0, 0.0, 0.0, 0.0]), q, n_sims=2000, seed=0))
    assert P[0, 3] == 1.0                                       # el favorito que abandona siempre es último


def test_fit_pl_recovers_known_parameters():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(500):
        pw = rng.dirichlet(np.ones(12)); ef = rng.uniform(2, 16, 12)
        s = log_strength(pw, ef, 0.8, 0.25) + rng.gumbel(size=12)
        for d, pos in enumerate(np.argsort(np.argsort(-s)) + 1):
            rows.append(dict(race_id=r, pw=pw[d], ef=ef[d], finish_position=pos, classified=True))
    a, b = fit_pl(pd.DataFrame(rows))
    assert abs(a - 0.8) < 0.1 and abs(b - 0.25) < 0.05


def test_summarize_interval_and_probabilities():
    P = position_matrix(simulate_positions(np.log(np.array([8.0, 1.0, 1.0, 1.0, 1.0])), np.zeros(5), n_sims=20_000, seed=3))
    S = summarize(P)
    assert (S["pos_lo"] <= S["exp_pos"]).all() and (S["exp_pos"] <= S["pos_hi"]).all() and np.isclose(S["p_win"].sum(), 1)


def _toy(seasons=range(2018, 2023)):
    raw = raw_toy(seasons=seasons, races=10)
    df = build_features(raw)
    rng = np.random.default_rng(0)
    oos = df[df["season"] >= 2020].copy()
    oos["pw"] = oos.groupby("race_id")["grid_eff"].transform(lambda g: np.exp(-g) / np.exp(-g).sum())
    oos["pp"] = np.clip(oos["pw"] * 3, 0, 1)
    oos["ef"] = oos["grid_eff"] + rng.normal(0, 0.5, len(oos))
    return raw, df, oos


def test_season_parameters_use_only_past_seasons():
    """Fuga: cambiar resultados y abandonos de la temporada evaluada no puede cambiar sus simulaciones."""
    raw, df, oos = _toy()
    cl = raw[["race_id", "driver_id", "classified"]]
    base = simulate_season(oos, df, cl, 2022, n_sims=2000)
    rng = np.random.default_rng(4)
    df2, oos2, cl2 = df.copy(), oos.copy(), cl.copy()
    for d in (df2, oos2):
        m = d["season"] == 2022
        d.loc[m, "target_finish"] = rng.permutation(d.loc[m, "target_finish"].to_numpy())
    m = cl2["race_id"].str[:4] == "2022"
    cl2.loc[m, "classified"] = ~cl2.loc[m, "classified"]
    pert = simulate_season(oos2, df2, cl2, 2022, n_sims=2000)
    assert len(base) == len(pert) == 10 and all(np.array_equal(a[2], b[2]) for a, b in zip(base, pert))


def test_evaluate_simulator_table():
    raw, df, oos = _toy()
    table, per_race = evaluate_simulator(oos, df, raw[["race_id", "driver_id", "classified"]], 2021, n_sims=2000)
    assert {"simulador PL", "simulador ajustado a P(win)", "grid empírica", "uniforme", "p_win/podio: simulador PL",
            "p_win/podio: simulador ajustado", "p_win/podio: modelo directo", "p_win/podio: promedio simulador-directo",
            "abandonos: modelo", "abandonos: tasa base"} <= set(table.index)
    assert 0 <= table.loc["simulador ajustado a P(win)", "cob80_acaba"] <= 1
    assert table.loc["uniforme", "log_score"] < 0 and 0 <= table.loc["simulador PL", "cobertura80"] <= 1




def test_match_win_reproduces_target_probabilities_with_retirements():
    from simulation.position_sim import match_win
    rng = np.random.default_rng(2)
    target = rng.dirichlet(np.ones(10) * 0.7)
    q = rng.uniform(0.02, 0.2, 10)
    s = match_win(np.zeros(10), q, target, n_sims=20_000, seed=0, iters=15)
    got = (simulate_positions(s, q, n_sims=60_000, seed=99) == 0).mean(axis=0)     # semilla distinta a la del ajuste
    assert np.abs(got - target).max() < 0.012





def test_conditional_matrix_and_same_randomness_as_simulate_positions():
    from simulation.position_sim import position_matrix_conditional, simulate_with_dnf
    s, q = np.log(np.array([5.0, 3.0, 1.0, 1.0])), np.array([0.3, 0.0, 0.0, 0.0])
    pos, dnf = simulate_with_dnf(s, q, n_sims=30_000, seed=5)
    assert np.array_equal(pos, simulate_positions(s, q, n_sims=30_000, seed=5))          # mismos números aleatorios
    Pc = position_matrix_conditional(pos, dnf)
    assert np.allclose(Pc.sum(axis=1), 1) and abs(dnf[:, 0].mean() - 0.3) < 0.01
    P = position_matrix(pos)                  # los demás no abandonan: quien abandona (piloto 0) queda siempre último
    assert np.isclose(P[0, 3], dnf[:, 0].mean() + (1 - dnf[:, 0].mean()) * Pc[0, 3], atol=1e-12)
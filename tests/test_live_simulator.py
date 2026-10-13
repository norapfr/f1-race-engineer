import numpy as np, pandas as pd
from features.build import build_features
from simulation.live import RaceSimulator
from tests.test_features import raw_toy
from tests.test_position_sim import _toy


def test_race_simulator_gives_coherent_summary_and_is_reproducible():
    raw, df, oos = _toy()
    cl = raw[["race_id", "driver_id", "classified"]]
    hist = df[df["season"] < 2022]
    sim = RaceSimulator.fit(oos[oos["season"] < 2022], hist, cl)
    g = oos[oos["race_id"] == "2022_05"]
    pred = g[["driver_id", "pw", "ef"]].rename(columns={"pw": "p_win", "ef": "exp_finish"})
    out = sim.run(pred, df[df["race_id"] == "2022_05"], n_sims=20_000, seed=3)
    assert len(out) == 6 and np.isclose(out["p_win"].sum(), 1.0, atol=1e-6)
    assert np.abs(out["p_win"].to_numpy() - pred["p_win"].to_numpy() / pred["p_win"].sum()).max() < 0.03   # ajustado a P(win)
    assert ((out["lo80"] <= out["lo50"]) & (out["lo50"] <= out["hi50"]) & (out["hi50"] <= out["hi80"])).all()
    assert out["p_dnf"].between(0, 1).all() and (out["p_podium"] >= out["p_win"] - 1e-9).all()
    assert (out["exp_pos_acaba"] <= out["exp_pos"] + 1e-9).all()      # contando solo cuando acaba, la posición media es mejor
    pd.testing.assert_frame_equal(out, sim.run(pred, df[df["race_id"] == "2022_05"], n_sims=20_000, seed=3))


def test_simulate_historical_uses_only_earlier_information():
    from simulation.live import simulate_historical
    raw, df, oos = _toy()
    cl = raw[["race_id", "driver_id", "classified"]]
    g = oos[oos["race_id"] == "2022_05"]
    pred = g[["driver_id", "pw", "ef"]].rename(columns={"pw": "p_win", "ef": "exp_finish"})
    base = simulate_historical(df, cl, oos, pred, "2022_05", n_sims=5000)
    df2, oos2, cl2 = df.copy(), oos.copy(), cl.copy()
    later = (df2["season"] * 100 + df2["round"]) >= 202205
    rng = np.random.default_rng(8)
    df2.loc[later, "target_finish"] = rng.permutation(df2.loc[later, "target_finish"].to_numpy())
    laterc = (oos2["season"] * 100 + oos2["round"]) >= 202205
    oos2.loc[laterc, "target_finish"] = rng.permutation(oos2.loc[laterc, "target_finish"].to_numpy())
    cl2["classified"] = np.where(cl2["race_id"] >= "2022_05", ~cl2["classified"], cl2["classified"])
    pd.testing.assert_frame_equal(base, simulate_historical(df2, cl2, oos2, pred, "2022_05", n_sims=5000))
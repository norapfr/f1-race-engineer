"""Evaluación del simulador con validación temporal: los parámetros de la temporada t se ajustan solo con temporadas < t."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.build import FEATURES
from models.race_model import FAMILIES, predict_season
from simulation.position_sim import (N_SIMS, DnfModel, fit_pl, log_strength, match_win, position_matrix,
                                     position_matrix_conditional, simulate_with_dnf, summarize)

FLOOR = 1e-4   # probabilidad mínima en el log score (1 de cada 10.000 simulaciones)


def oos_predictions(df: pd.DataFrame, min_train_seasons: int = 3) -> pd.DataFrame:
    """Predicciones fuera de muestra del conjunto (logística + LightGBM, todas las features) para cada temporada de test."""
    seasons, parts = sorted(df["season"].unique()), []
    for t in seasons[min_train_seasons:]:
        outs = [predict_season(fam, df, t, FEATURES) for fam in FAMILIES]
        te = outs[0][0].copy()
        for col, key in (("pw", "p_win"), ("pp", "p_pod"), ("ef", "ef")):
            te[col] = np.mean([o[1][key] for o in outs], axis=0)
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def _with_classified(d: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    m = d.merge(classified, on=["race_id", "driver_id"], how="left")
    m["classified"] = m["classified"].fillna(True).astype(bool)
    return m.assign(finish_position=m["target_finish"])


def slot_distribution(train: pd.DataFrame, n_max: int = 22, alpha: float = 0.5) -> np.ndarray:
    """Baseline: distribución empírica de la posición final según la posición de salida (suavizada)."""
    M = np.full((n_max, n_max), alpha)
    slot = np.clip(train["grid_eff"].to_numpy().round().astype(int), 1, n_max) - 1
    pos = np.clip(train["target_finish"].to_numpy().round().astype(int), 1, n_max) - 1
    np.add.at(M, (slot, pos), 1.0)
    return M / M.sum(axis=1, keepdims=True)


def _baseline_P(g: pd.DataFrame, M: np.ndarray) -> np.ndarray:
    n = len(g)
    slot = np.clip(g["grid_eff"].to_numpy().round().astype(int), 1, M.shape[0]) - 1
    P = M[slot][:, :n]
    return P / P.sum(axis=1, keepdims=True)


def simulate_season(oos: pd.DataFrame, df: pd.DataFrame, classified: pd.DataFrame, t: int,
                    n_sims: int = N_SIMS, seed: int = 0, match: bool = False) -> list:
    """Para cada carrera de la temporada t: (race_id, filas, P, Pc, q) con P = posición final, Pc = posición final si el
    piloto acaba y q = P(abandono). Parámetros ajustados SOLO con temporadas < t.
    match=True: las fuerzas se ajustan para reproducir las P(victoria) del modelo."""
    a, b = fit_pl(_with_classified(oos[oos["season"] < t], classified))
    dnf = DnfModel().fit(_with_classified(df[df["season"] < t], classified))
    out = []
    for i, (race_id, g) in enumerate(oos[oos["season"] == t].groupby("race_id", sort=True)):
        g = g.reset_index(drop=True)
        s0, q = log_strength(g["pw"], g["ef"], a, b), dnf.predict(g)
        if match:
            s0 = match_win(s0, q, g["pw"].to_numpy(), n_sims, seed + i)
            pos, dnf_draw = simulate_with_dnf(s0, q, n_sims, seed + i + 100_000)   # semilla distinta a la del ajuste
        else:
            pos, dnf_draw = simulate_with_dnf(s0, q, n_sims, seed + i)
        out.append((race_id, g, position_matrix(pos), position_matrix_conditional(pos, dnf_draw), q))
    return out


def _interval(Q: np.ndarray, level: float):
    cum, tail = np.cumsum(Q, axis=1), (1 - level) / 2
    return (cum >= tail).argmax(axis=1) + 1, (cum >= 1 - tail - 1e-12).argmax(axis=1) + 1


def _position_metrics(Q: np.ndarray, r: np.ndarray) -> dict:
    out = {"log_score": float(np.mean(np.log(np.maximum(Q[np.arange(len(r)), r - 1], FLOOR))))}
    for level in (0.5, 0.8):
        lo, hi = _interval(Q, level)
        k = int(level * 100)
        out[f"cobertura{k}"], out[f"ancho{k}"] = float(np.mean((lo <= r) & (r <= hi))), float(np.mean(hi - lo + 1))
    return out


def _marginals(P: np.ndarray, pw, pp, r: np.ndarray) -> dict:
    S = summarize(P)
    yw, yp = (r == 1).astype(float), (r <= 3).astype(float)
    ps = np.clip(S["p_win"].to_numpy(), 1e-6, 1 - 1e-6)
    return {"logloss_win": float(-np.mean(yw * np.log(ps) + (1 - yw) * np.log(1 - ps))),
            "brier_podio": float(np.mean((S["p_podium"].to_numpy() - yp) ** 2))}


def _finisher_metrics(Pc: np.ndarray, r: np.ndarray, fin: np.ndarray) -> dict:
    """Intervalos condicionados a acabar, evaluados solo sobre quienes acabaron."""
    out = {}
    for level in (0.5, 0.8):
        lo, hi = _interval(Pc, level)
        k = int(level * 100)
        out[f"cob{k}_acaba"], out[f"ancho{k}_acaba"] = float(np.mean(((lo <= r) & (r <= hi))[fin])), float(np.mean((hi - lo + 1)[fin]))
    return out


def evaluate_simulator(oos: pd.DataFrame, df: pd.DataFrame, classified: pd.DataFrame, first_test: int,
                       n_sims: int = N_SIMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for t in sorted(oos["season"].unique()):
        if t < first_test:
            continue
        M = slot_distribution(df[df["season"] < t])
        base_dnf = float((~_with_classified(df[df["season"] < t], classified)["classified"]).mean())
        pl = simulate_season(oos, df, classified, t, n_sims)
        adj = simulate_season(oos, df, classified, t, n_sims, match=True)
        for (race_id, g, P1, Pc1, q), (_, _, P2, Pc2, _) in zip(pl, adj):
            n, r = len(g), g["target_finish"].to_numpy().astype(int)
            fin = _with_classified(g, classified)["classified"].to_numpy()
            yw, yp = (r == 1).astype(float), (r <= 3).astype(float)
            base = {"season": t, "race_id": race_id}
            for name, Q, Qc in (("simulador PL", P1, Pc1), ("simulador ajustado a P(win)", P2, Pc2)):
                rows.append({**base, "modelo": name, **_position_metrics(Q, r), **_finisher_metrics(Qc, r, fin)})
            for name, Q in (("grid empírica", _baseline_P(g, M)), ("uniforme", np.full((n, n), 1 / n))):
                rows.append({**base, "modelo": name, **_position_metrics(Q, r)})
            pw, pp = np.clip(g["pw"].to_numpy(), 1e-6, 1 - 1e-6), g["pp"].to_numpy()
            for name, P in (("p_win/podio: simulador PL", P1), ("p_win/podio: simulador ajustado", P2)):
                rows.append({**base, "modelo": name, **_marginals(P, pw, pp, r)})
            rows.append({**base, "modelo": "p_win/podio: modelo directo",
                         "logloss_win": float(-np.mean(yw * np.log(pw) + (1 - yw) * np.log(1 - pw))),
                         "brier_podio": float(np.mean((pp - yp) ** 2))})
            pod_avg = 0.5 * (summarize(P2)["p_podium"].to_numpy() + pp)
            rows.append({**base, "modelo": "p_win/podio: promedio simulador-directo", "brier_podio": float(np.mean((pod_avg - yp) ** 2))})
            dnf_real = (~fin).astype(float)
            rows.append({**base, "modelo": "abandonos: modelo", "brier_dnf": float(np.mean((q - dnf_real) ** 2)),
                         "dnf_predicho": float(q.mean()), "dnf_real": float(dnf_real.mean())})
            rows.append({**base, "modelo": "abandonos: tasa base", "brier_dnf": float(np.mean((base_dnf - dnf_real) ** 2)),
                         "dnf_predicho": base_dnf, "dnf_real": float(dnf_real.mean())})
    d = pd.DataFrame(rows)
    return d.groupby("modelo").mean(numeric_only=True).drop(columns="season"), d
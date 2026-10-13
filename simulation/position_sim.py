"""Simulador Monte Carlo de la clasificación final de una carrera.
Cada piloto tiene una 'fuerza' log s = a*log(P(win)) - b*(posición esperada), construida con las salidas del modelo.
Los pilotos que acaban se ordenan por s + ruido Gumbel (equivale a un modelo de Plackett-Luce); los que abandonan
(probabilidad q, estimada con su historial de abandonos y el de su equipo) quedan detrás de todos los que acaban.
Con la misma semilla el resultado es idéntico. Un parámetro de cada tipo (a, b) se ajusta por máxima verosimilitud."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

N_SIMS = 10_000
DNF_COLS = ["prior_driver_dnf_ewm", "prior_team_dnf_ewm"]


def log_strength(pw, ef, a: float, b: float) -> np.ndarray:
    return a * np.log(np.clip(np.asarray(pw, float), 1e-6, 1.0)) - b * np.asarray(ef, float)


def fit_pl(oos: pd.DataFrame) -> tuple[float, float]:
    """Máxima verosimilitud de Plackett-Luce sobre el orden de los que acaban la carrera.
    oos: race_id, pw, ef, finish_position, classified (predicciones fuera de muestra)."""
    races = []
    for _, g in oos[oos["classified"].astype(bool)].groupby("race_id"):
        g = g.sort_values("finish_position")
        if len(g) >= 2:
            races.append((np.log(np.clip(g["pw"].to_numpy(float), 1e-6, 1.0)), -g["ef"].to_numpy(float)))

    def nll(p):
        tot = 0.0
        for lp, nef in races:
            s = p[0] * lp + p[1] * nef
            tot -= float(np.sum(s - np.logaddexp.accumulate(s[::-1])[::-1]))
        return tot

    return tuple(float(v) for v in minimize(nll, x0=[0.5, 0.2], method="L-BFGS-B").x)


class DnfModel:
    """P(no acabar) con el historial de abandonos del piloto y del equipo (solo carreras previas)."""
    def fit(self, d: pd.DataFrame):
        dnf = (~d["classified"].astype(bool)).to_numpy()
        self.med = d[DNF_COLS].median().fillna(0.0)
        self.const = float(dnf.mean()) if dnf.min() == dnf.max() else None
        if self.const is None:
            self.lr = LogisticRegression(C=1.0, max_iter=1000).fit(d[DNF_COLS].fillna(self.med), dnf)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        if self.const is not None:
            return np.full(len(d), self.const)
        return self.lr.predict_proba(d[DNF_COLS].fillna(self.med))[:, 1]


def simulate_positions(s: np.ndarray, q: np.ndarray, n_sims: int = N_SIMS, seed: int = 0) -> np.ndarray:
    """Devuelve (n_sims, n): posición final (0 = ganador) de cada piloto en cada carrera simulada."""
    rng = np.random.default_rng(seed)
    n = len(s)
    key = np.asarray(s, float)[None, :] + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < np.asarray(q, float)[None, :]
    key = np.where(dnf, -1e6 + rng.random((n_sims, n)), key)       # los que abandonan, detrás y en orden aleatorio
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n_sims)[:, None], order] = np.arange(n)[None, :]
    return pos


def position_matrix(pos: np.ndarray) -> np.ndarray:
    """P[d, k] = probabilidad de que el piloto d termine en la posición k+1."""
    n_sims, n = pos.shape
    return np.stack([np.bincount(pos[:, d], minlength=n) for d in range(n)]) / n_sims


def summarize(P: np.ndarray, level: float = 0.90) -> pd.DataFrame:
    n = P.shape[1]
    cum, tail = P.cumsum(axis=1), (1 - level) / 2
    return pd.DataFrame({"exp_pos": P @ np.arange(1, n + 1),
                         "pos_lo": (cum >= tail).argmax(axis=1) + 1, "pos_hi": (cum >= 1 - tail - 1e-12).argmax(axis=1) + 1,
                         "p_win": P[:, 0], "p_podium": P[:, :3].sum(axis=1),
                         "p_top5": P[:, :5].sum(axis=1), "p_top10": P[:, :10].sum(axis=1)})



def match_win(s0, q, target_pw, n_sims: int = N_SIMS, seed: int = 0, iters: int = 10, damping: float = 0.9) -> np.ndarray:
    """Ajusta la fuerza de cada piloto para que el simulador (con abandonos incluidos) reproduzca sus probabilidades de
    victoria objetivo. Se usan los mismos números aleatorios en cada iteración, y el resultado se valida con otra semilla."""
    s = np.asarray(s0, float).copy()
    target = np.clip(np.asarray(target_pw, float), 1e-4, None)
    target = target / target.sum()
    for _ in range(iters):
        got = np.clip((simulate_positions(s, q, n_sims, seed) == 0).mean(axis=0), 1e-4, None)
        s += np.clip(damping * (np.log(target) - np.log(got)), -2.0, 2.0)
    return s




def simulate_with_dnf(s: np.ndarray, q: np.ndarray, n_sims: int = N_SIMS, seed: int = 0):
    """Igual que simulate_positions (mismos números aleatorios) pero devuelve también quién abandona en cada carrera."""
    rng = np.random.default_rng(seed)
    n = len(s)
    key = np.asarray(s, float)[None, :] + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < np.asarray(q, float)[None, :]
    key = np.where(dnf, -1e6 + rng.random((n_sims, n)), key)
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n_sims)[:, None], order] = np.arange(n)[None, :]
    return pos, dnf


def position_matrix_conditional(pos: np.ndarray, dnf: np.ndarray) -> np.ndarray:
    """P[d, k] = probabilidad de terminar en la posición k+1 SI el piloto d acaba la carrera (descarta sus abandonos)."""
    n_sims, n = pos.shape
    out = np.full((n, n), 1.0 / n)
    for d in range(n):
        ok = ~dnf[:, d]
        if ok.any():
            out[d] = np.bincount(pos[ok, d], minlength=n) / ok.sum()
    return out
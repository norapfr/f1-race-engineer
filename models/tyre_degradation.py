"""Modelo de degradación de neumáticos por efectos fijos (dos vías).
    tiempo(carrera, piloto, vuelta) = EF(carrera, vuelta) + EF(carrera, piloto) + nivel(compuesto) + g_compuesto(edad)
- EF(carrera, vuelta): todo lo común a esa vuelta (combustible, evolución de la pista, ritmo del grupo).
- EF(carrera, piloto): ritmo propio de cada piloto en esa carrera.
- g_compuesto(edad): curva lineal a tramos (nudos en KNOTS); se estima por la variación entre pilotos en la MISMA vuelta
  con distinta edad/compuesto. El combustible no hace falta separarlo: al comparar estrategias de un mismo piloto
  en las mismas vueltas, cancela. Errores estándar robustos por carrera."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
REF = "MEDIUM"                     # su nivel base es 0; los demás se miden frente a él
KNOTS = (5, 10, 15, 20, 30)


def _basis(ok: pd.DataFrame):
    age = ok["tyre_life"].to_numpy(float)
    cols, names = [], []
    for c in COMPOUNDS:
        m = (ok["compound"] == c).to_numpy(float)
        if c != REF:
            cols.append(m); names.append(f"off_{c}")
        cols.append(m * age); names.append(f"slope_{c}")
        for k in KNOTS:
            cols.append(m * np.maximum(0.0, age - k)); names.append(f"hinge{k}_{c}")
    return np.column_stack(cols), names


def demean(M: np.ndarray, groups: list[np.ndarray], iters: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Quita las medias de varios grupos a la vez (proyecciones alternas): equivale a incluir efectos fijos."""
    X = M.astype(float).copy()
    sizes = [np.bincount(g) for g in groups]
    for _ in range(iters):
        delta = 0.0
        for g, cnt in zip(groups, sizes):
            for j in range(X.shape[1]):
                m = (np.bincount(g, weights=X[:, j], minlength=len(cnt)) / cnt)[g]
                X[:, j] -= m
                delta = max(delta, float(np.abs(m).max()))
        if delta < tol:
            break
    return X


@dataclass
class DegFit:
    beta: pd.Series
    cov: pd.DataFrame
    n_laps: int
    n_races: int

    def _vec(self, spec: dict) -> np.ndarray:
        v = pd.Series(0.0, index=self.beta.index)
        for k, w in spec.items():
            if k in v.index:
                v[k] += w
        return v.to_numpy()

    def _est(self, spec: dict):
        v = self._vec(spec)
        return float(v @ self.beta.to_numpy()), float(np.sqrt(max(v @ self.cov.to_numpy() @ v, 0.0)))

    def loss(self, compound: str, age: float):
        """Tiempo que pierde un neumático de `age` vueltas frente a uno de 1 vuelta (s) y su error estándar."""
        spec = {f"slope_{compound}": age - 1.0}
        spec.update({f"hinge{k}_{compound}": max(0.0, age - k) for k in KNOTS})
        return self._est(spec)

    def fresh_offset(self, compound: str):
        """Nivel de un neumático nuevo (1 vuelta) de `compound` frente al medio nuevo (s; negativo = más rápido)."""
        spec = {f"slope_{compound}": 1.0, f"slope_{REF}": -1.0}
        if compound != REF:
            spec[f"off_{compound}"] = 1.0
        return self._est(spec)


def fit_degradation(ok: pd.DataFrame) -> DegFit:
    """`ok`: vueltas limpias en seco (features.degradation.clean_dry_laps)."""
    lap_g = pd.factorize(ok["race_id"] + "|" + ok["lap_number"].astype(str))[0]
    drv_g = pd.factorize(ok["race_id"] + "|" + ok["driver_id"])[0]
    B, names = _basis(ok)
    Z = demean(np.column_stack([ok["lap_time_s"].to_numpy(float), B]), [lap_g, drv_g])
    y, X = Z[:, 0], Z[:, 1:]
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    cl = pd.factorize(ok["race_id"])[0]
    S = np.zeros((cl.max() + 1, X.shape[1]))
    np.add.at(S, cl, X * e[:, None])
    cov = inv @ (S.T @ S) @ inv
    return DegFit(pd.Series(beta, index=names), pd.DataFrame(cov, index=names, columns=names), len(ok), int(cl.max() + 1))


def curve_tables(fit: DegFit, ages=(5, 10, 15, 20, 30)):
    est, se = {}, {}
    for c in COMPOUNDS:
        est[c], se[c] = zip(*[fit.loss(c, a) for a in ages])
    fmt = lambda e, s: pd.DataFrame({c: [f"{a:+.2f}±{b:.2f}" for a, b in zip(e[c], s[c])] for c in COMPOUNDS}, index=list(ages))
    return fmt(est, se), {c: fit.fresh_offset(c) for c in COMPOUNDS}
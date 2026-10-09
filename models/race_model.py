"""Modelos de predicción de carrera: P(win), P(podio), P(top5), P(top10) y posición esperada.
Familias: regresión logística regularizada y LightGBM con regularización fuerte.
Hiperparámetros elegidos con la temporada INMEDIATAMENTE anterior a la evaluada (nunca con la evaluada)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.build import FEATURES

# clave -> (columna objetivo, nº de plazas que suma por carrera)
TARGETS_K = {"p_win": ("target_win", 1), "p_pod": ("target_podium", 3),
             "p_top5": ("target_top5", 5), "p_top10": ("target_top10", 10)}


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def normalize_by_race(p: np.ndarray, race_ids: np.ndarray, k: int) -> np.ndarray:
    """Fuerza que las probabilidades de cada carrera sumen k (1 ganador, 3 podio...), manteniendo p <= 1."""
    p = np.asarray(p, float)
    out = np.empty_like(p)
    for _, idx in pd.Series(np.arange(len(p))).groupby(race_ids).groups.items():
        idx = np.asarray(idx)
        q, kk = np.clip(p[idx], 1e-6, 1 - 1e-6), min(k, len(idx) - 1e-3)
        for _ in range(30):
            q = np.clip(q * kk / q.sum(), 1e-6, 1 - 1e-6)
        out[idx] = q
    return out


def _prep():
    return [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]

SEEDS = (0, 1, 2)


class SeedAvg:
    """Promedia varios modelos con distinta semilla de bagging: reduce el ruido del submuestreo aleatorio."""
    def __init__(self, make):
        self.make = make

    def fit(self, X, y):
        self.models = [self.make(s).fit(X, y) for s in SEEDS]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    
class Logit:
    name = "logit"
    candidates = [{"C": 0.03}, {"C": 0.3}, {"C": 3.0}]

    def make_clf(self, cfg):
        return Pipeline(_prep() + [("m", LogisticRegression(C=cfg["C"], max_iter=3000))])

    def make_reg(self, cfg):
        return Pipeline(_prep() + [("m", Ridge(alpha=1.0 / cfg["C"]))])


class GBM:
    name = "lgbm"
    candidates = [dict(num_leaves=3, n_estimators=150), dict(num_leaves=6, n_estimators=150),
                  dict(num_leaves=6, n_estimators=300)]
    common = dict(learning_rate=0.03, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=0, n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True)

    def make_clf(self, cfg):
        return SeedAvg(lambda s: LGBMClassifier(**{**self.common, "random_state": s}, **cfg))
    
    def make_reg(self, cfg):
        return SeedAvg(lambda s: LGBMRegressor(**{**self.common, "random_state": s}, **cfg))


FAMILIES = [Logit(), GBM()]


def fit_predict(fam, cfg: dict, tr: pd.DataFrame, te: pd.DataFrame, features=None) -> dict:
    features = list(features or FEATURES)
    out = {}
    for key, (target, k) in TARGETS_K.items():
        if tr[target].nunique() < 2:   # objetivo degenerado (una sola clase): probabilidad constante
            raw = np.full(len(te), float(tr[target].mean()))
        else:
            raw = fam.make_clf(cfg).fit(tr[features], tr[target]).predict_proba(te[features])[:, 1]
        out[key] = normalize_by_race(raw, te["race_id"].to_numpy(), k)
    out["ef"] = fam.make_reg(cfg).fit(tr[features], tr["target_finish"]).predict(te[features])
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt_fit(p, y):
    """Escalado de Platt (2 parámetros sobre el logit). None si no hay ambas clases."""
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(C=1.0, max_iter=1000).fit(_logit(p), y)


def platt_apply(cal, p) -> np.ndarray:
    return np.asarray(p, float) if cal is None else cal.predict_proba(_logit(p))[:, 1]


def recalibrate(P: dict, te: pd.DataFrame) -> dict:
    """Recalibra P(win) y P(podio) con las predicciones fuera de muestra de la temporada t-1 (ya conocidas)."""
    out = {}
    for key in ("p_win", "p_pod"):
        cal = platt_fit(P[f"val_{key}"], P[f"val_y_{key}"])
        out[key] = normalize_by_race(platt_apply(cal, P[key]), te["race_id"].to_numpy(), TARGETS_K[key][1])
    return out


def predict_season(fam, df: pd.DataFrame, t: int, features=None):
    """Predice la temporada t usando SOLO temporadas < t. La configuración se elige con la temporada t-1
    (entrenando con < t-1). Devuelve (te, predicciones, configuración elegida)."""
    val = max(s for s in df["season"].unique() if s < t)
    tr_v, va = df[df["season"] < val], df[df["season"] == val].reset_index(drop=True)
    best, best_score, best_P = None, np.inf, None
    for cfg in fam.candidates:
        P = fit_predict(fam, cfg, tr_v, va, features)
        score = logloss(va["target_win"], P["p_win"]) + logloss(va["target_podium"], P["p_pod"])
        if score < best_score:
            best, best_score, best_P = cfg, score, P
    tr, te = df[df["season"] < t], df[df["season"] == t].reset_index(drop=True)
    P = fit_predict(fam, best, tr, te, features)
    # predicciones fuera de muestra de la temporada t-1: sirven para recalibrar (solo información pasada)
    P.update(val_p_win=best_P["p_win"], val_y_p_win=va["target_win"].to_numpy(float),
             val_p_pod=best_P["p_pod"], val_y_p_pod=va["target_podium"].to_numpy(float))
    return te, P, best
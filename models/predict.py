"""Predicción de una carrera concreta con el modelo de referencia `ens`
(promedio de regresión logística y LightGBM, ambos con todas las features).
Entrena SOLO con carreras anteriores a la predicha; los hiperparámetros se eligen con la temporada previa."""
from __future__ import annotations
import pandas as pd
from features.build import FEATURES
from models.race_model import FAMILIES, fit_predict, logloss

REFERENCE_MODEL = "ens"


def _select_config(fam, df: pd.DataFrame, season: int, features) -> dict:
    """Elige la configuración con la temporada anterior (entrenando con las previas a esa)."""
    val = season - 1
    tr_v, va = df[df["season"] < val], df[df["season"] == val].reset_index(drop=True)
    scores = []
    for i, cfg in enumerate(fam.candidates):
        P = fit_predict(fam, cfg, tr_v, va, features)
        scores.append((logloss(va["target_win"], P["p_win"]) + logloss(va["target_podium"], P["p_pod"]), i))
    return fam.candidates[min(scores)[1]]


def predict_race(df: pd.DataFrame, race_id: str, features=None) -> pd.DataFrame:
    """Probabilidades por piloto: p_win, p_podium, p_top5, p_top10 y posición esperada."""
    features = list(features or FEATURES)
    te = df[df["race_id"] == race_id].reset_index(drop=True)
    if te.empty:
        raise ValueError(f"Carrera desconocida: {race_id}")
    season, rnd = int(te["season"].iloc[0]), int(te["round"].iloc[0])
    past = df[(df["season"] * 100 + df["round"]) < season * 100 + rnd]
    seasons = set(past["season"].unique())
    if not {season - 1, season - 2} <= seasons:
        raise ValueError("Hacen falta al menos dos temporadas completas anteriores a la carrera.")
    outs = []
    for fam in FAMILIES:
        cfg = _select_config(fam, past, season, features)
        outs.append(fit_predict(fam, cfg, past, te, features))
    avg = lambda k: sum(o[k] for o in outs) / len(outs)
    res = te[["race_id", "driver_id", "team_id", "grid_eff"]].copy()
    res["p_win"], res["p_podium"], res["p_top5"], res["p_top10"] = avg("p_win"), avg("p_pod"), avg("p_top5"), avg("p_top10")
    res["exp_finish"] = avg("ef")
    return res.sort_values("p_win", ascending=False).reset_index(drop=True)
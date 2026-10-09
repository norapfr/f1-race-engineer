"""Calibración (curva, ECE) e intervalos por bootstrap de carreras."""
from __future__ import annotations
import numpy as np
import pandas as pd

BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0001]


def calibration_table(y, p) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.cut(d["p"], BINS, right=False)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), pred_media=("p", "mean"), frec_obs=("y", "mean"))
    return g.reset_index()


def ece(y, p) -> float:
    t = calibration_table(y, p)
    return float((t["n"] * (t["pred_media"] - t["frec_obs"]).abs()).sum() / t["n"].sum())


def race_terms(te: pd.DataFrame, pw, ef) -> pd.DataFrame:
    """Suma por carrera de la pérdida logarítmica de victoria y del error absoluto de posición."""
    yw = (te["finish_position"] == 1).to_numpy(float)
    p = np.clip(np.asarray(pw, float), 1e-6, 1 - 1e-6)
    d = pd.DataFrame({"race_id": te["race_id"].to_numpy(), "ll": -(yw * np.log(p) + (1 - yw) * np.log(1 - p)),
                      "ae": np.abs(np.asarray(ef, float) - te["finish_position"].to_numpy(float))})
    return d.groupby("race_id").agg(ll=("ll", "sum"), ae=("ae", "sum"), n=("ll", "size")).sort_index()


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0):
    """Diferencia media (a - b) de la métrica `col` remuestreando CARRERAS. Devuelve (media, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    num = a[col].to_numpy()[idx].sum(1) - b[col].to_numpy()[idx].sum(1)
    d = num / a["n"].to_numpy()[idx].sum(1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
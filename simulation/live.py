"""Simulación de una carrera concreta con parámetros ajustados con todo el historial disponible.
Mismo procedimiento que en la evaluación (simulador ajustado a P(win)), con 10.000 carreras simuladas y semilla fija."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from simulation.evaluate import _interval, _with_classified, oos_predictions
from simulation.position_sim import (DNF_COLS, N_SIMS, DnfModel, fit_pl, log_strength, match_win, position_matrix,
                                     position_matrix_conditional, simulate_with_dnf, summarize)

OOS_PATH = Path("data/processed/oos_ens.parquet")


def load_oos(hist_df: pd.DataFrame, path: Path = OOS_PATH, refresh: bool = False) -> pd.DataFrame:
    """Predicciones fuera de muestra del conjunto (para ajustar el simulador). Se cachean en disco."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    oos = oos_predictions(hist_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    oos.to_parquet(path)
    return oos


class RaceSimulator:
    def __init__(self, a: float, b: float, dnf: DnfModel):
        self.a, self.b, self.dnf = a, b, dnf

    @classmethod
    def fit(cls, oos: pd.DataFrame, hist_df: pd.DataFrame, classified: pd.DataFrame) -> "RaceSimulator":
        """oos y hist_df: SOLO carreras con resultado conocido. classified: race_id, driver_id, classified."""
        a, b = fit_pl(_with_classified(oos, classified))
        return cls(a, b, DnfModel().fit(_with_classified(hist_df, classified)))

    def run(self, pred: pd.DataFrame, feats: pd.DataFrame, n_sims: int = N_SIMS, seed: int = 0) -> pd.DataFrame:
        """pred: driver_id, p_win, exp_finish (salida de predict_race). feats: filas de la carrera con las features de abandono.
        Devuelve por piloto: posición esperada, P(victoria/podio/top5/top10), P(abandono) e intervalos al 50% y 80% de la
        posición final CONDICIONADOS a que el piloto acabe (lo50..hi80); exp_pos_acaba = posición esperada si acaba."""
        d = pred[["driver_id", "p_win", "exp_finish"]].merge(feats[["driver_id"] + DNF_COLS], on="driver_id", how="left")
        q = self.dnf.predict(d)
        s = match_win(log_strength(d["p_win"], d["exp_finish"], self.a, self.b), q, d["p_win"].to_numpy(), n_sims, seed)
        pos, dnf = simulate_with_dnf(s, q, n_sims, seed + 100_000)               # semilla distinta a la del ajuste
        P, Pc = position_matrix(pos), position_matrix_conditional(pos, dnf)
        S = summarize(P)
        out = pd.DataFrame({"driver_id": d["driver_id"], "exp_pos": S["exp_pos"], "p_win": S["p_win"],
                            "p_podium": S["p_podium"], "p_top5": S["p_top5"], "p_top10": S["p_top10"], "p_dnf": q,
                            "exp_pos_acaba": Pc @ np.arange(1, Pc.shape[1] + 1)})
        for level in (0.5, 0.8):
            lo, hi = _interval(Pc, level)
            out[f"lo{int(level * 100)}"], out[f"hi{int(level * 100)}"] = lo, hi
        return out


def simulate_historical(df: pd.DataFrame, classified: pd.DataFrame, oos_all: pd.DataFrame, pred: pd.DataFrame,
                        race_id: str, n_sims: int = N_SIMS, seed: int = 0) -> pd.DataFrame:
    """Simula una carrera YA disputada usando solo información anterior a ella (parámetros y abandonos)."""
    key = lambda d: d["season"] * 100 + d["round"]
    race = df[df["race_id"] == race_id]
    k = int(key(race).iloc[0])
    oos = oos_all[key(oos_all) < k]
    if oos["race_id"].nunique() < 5:
        raise ValueError("Hay muy pocas predicciones previas para ajustar el simulador (carrera demasiado temprana).")
    return RaceSimulator.fit(oos, df[key(df) < k], classified).run(pred, race, n_sims, seed)
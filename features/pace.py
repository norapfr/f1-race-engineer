"""Ritmo de carrera por piloto a partir de las vueltas de FastF1.
Es información POST_RACE: nunca se usa directamente, solo desplazada (prior_*) en features/build.py."""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_FIELD = 5          # mínimo de pilotos con vuelta limpia en una vuelta para usar su mediana como referencia
MIN_CLEAN_LAPS = 10    # mínimo de vueltas limpias de un piloto para fiarse de su ritmo
WET_SHARE = 0.2        # carrera "mojada" si >= 20% de las vueltas son con intermedios/lluvia
MAX_REL_PCT = 7.0      # descarta vueltas más de un 7% fuera de la mediana del campo
WET = {"INTERMEDIATE", "WET"}


def race_pace_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Ritmo relativo de cada piloto en cada carrera, en % frente a la mediana del campo en la misma vuelta.
    Negativo = más rápido que el campo. Solo vueltas limpias: sin vuelta 1, sin boxes, con bandera verde y precisas.
    Carreras mojadas o con pocas vueltas limpias (p. ej. tras safety car) quedan en NaN."""
    d = laps.copy()
    for c in ("is_accurate", "pit_in", "pit_out"):
        d[c] = d[c].fillna(False).astype(bool)
    wet = (d["compound"].isin(WET).groupby(d["race_id"]).mean() >= WET_SHARE).rename("race_wet").reset_index()
    ok = d[(d["lap_number"] > 1) & d["lap_time_s"].notna() & d["is_accurate"] & ~d["pit_in"] & ~d["pit_out"]
           & (d["track_status"] == "1")].copy()
    g = ok.groupby(["race_id", "lap_number"])["lap_time_s"]
    ok["med"], ok["cnt"] = g.transform("median"), g.transform("count")
    ok = ok[ok["cnt"] >= MIN_FIELD].copy()
    ok["rel"] = (ok["lap_time_s"] / ok["med"] - 1) * 100
    ok = ok[ok["rel"].abs() <= MAX_REL_PCT]
    out = (ok.groupby(["race_id", "driver_id"]).agg(race_pace_pct=("rel", "median"), n_clean_laps=("rel", "size"))
             .reset_index().merge(wet, on="race_id", how="left"))
    out["race_wet"] = out["race_wet"].fillna(False).astype(bool)
    out.loc[(out["n_clean_laps"] < MIN_CLEAN_LAPS) | out["race_wet"], "race_pace_pct"] = np.nan
    return out
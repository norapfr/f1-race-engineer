"""Predice una carrera con el modelo de referencia, usando solo carreras anteriores.
Uso: python -m pipelines.predict_race 2026_16 [data/f1.duckdb]"""
from __future__ import annotations
import sys
import pandas as pd
from features.build import build_features, load_raw
from models.predict import REFERENCE_MODEL, predict_race
from storage.db import connect

if __name__ == "__main__":
    race_id = sys.argv[1]
    con = connect(sys.argv[2] if len(sys.argv) > 2 else "data/f1.duckdb")
    df = build_features(load_raw(con))
    pred = predict_race(df, race_id)
    names = con.execute("select driver_id, name from clean.drivers").df()
    teams = con.execute("select team_id, name as equipo from clean.teams").df()
    real = df.loc[df["race_id"] == race_id, ["driver_id", "target_finish"]].rename(columns={"target_finish": "real"})
    out = pred.merge(names, on="driver_id", how="left").merge(teams, on="team_id", how="left").merge(real, on="driver_id", how="left")
    for c in ("p_win", "p_podium", "p_top5", "p_top10"):
        out[c] = (100 * out[c]).round(1)
    out["exp_finish"] = out["exp_finish"].round(1)
    title = con.execute("select name from clean.races where race_id = ?", [race_id]).fetchone()
    print(f"\n{race_id} — {title[0] if title else ''} (modelo: {REFERENCE_MODEL}; probabilidades en %)\n")
    print(out[["name", "equipo", "grid_eff", "p_win", "p_podium", "p_top5", "p_top10", "exp_finish", "real"]]
          .rename(columns={"name": "piloto", "grid_eff": "parrilla"}).to_string(index=False))
    print(f"\nSuma P(win) = {pred['p_win'].sum():.2f}; suma P(podio) = {pred['p_podium'].sum():.2f}")
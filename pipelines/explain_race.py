"""Explica por qué el modelo da esas probabilidades de victoria en una carrera ya disputada.
Uso: python -m pipelines.explain_race 2026_16 [--top 6] [--detail] [--db data/f1.duckdb]"""
from __future__ import annotations
import argparse
import pandas as pd
from features.build import build_features, load_raw
from models.explain import explain_groups, explain_race, explain_table
from models.predict import predict_race
from storage.db import connect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("race_id")
    ap.add_argument("--top", type=int, default=6, help="nº de pilotos favoritos a explicar")
    ap.add_argument("--db", default="data/f1.duckdb")
    ap.add_argument("--detail", action="store_true", help="muestra además los factores individuales")
    a = ap.parse_args()
    con = connect(a.db)
    df = build_features(load_raw(con))
    pred = predict_race(df, a.race_id).head(a.top)
    names = con.execute("select driver_id, name from clean.drivers").df().set_index("driver_id")["name"]
    C = explain_race(df, a.race_id)
    order = list(pred["driver_id"])
    groups = explain_groups(C, names, order)
    groups.insert(1, "P(win) %", (100 * pred["p_win"]).round(1).to_numpy())
    pd.set_option("display.width", 250)
    print(f"\n{a.race_id}: efecto de cada grupo de factores sobre las probabilidades de ganar "
          f"(x = multiplica las probabilidades frente a un piloto medio)\n")
    print(groups.round(2).to_string(index=False))
    if a.detail:
        pd.set_option("display.max_colwidth", 200)
        print("\nDetalle por factor individual (los correlacionados se compensan entre sí; mejor leer los grupos):\n")
        print(explain_table(C, names, order).to_string(index=False))
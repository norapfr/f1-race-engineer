"""Evalúa el simulador de posiciones con validación temporal (parámetros ajustados solo con temporadas anteriores).
Uso: python -m pipelines.eval_simulator data/f1.duckdb [--refresh]
Las predicciones fuera de muestra se guardan en data/processed/oos_ens.parquet (--refresh las recalcula)."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from features.build import build_features, load_raw
from simulation.evaluate import evaluate_simulator, oos_predictions
from storage.db import connect

FIRST_TEST = 2022   # 2021 sirve solo para ajustar los parámetros del simulador

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    con = connect(args[0] if args else "data/f1.duckdb")
    raw = load_raw(con)
    df = build_features(raw)
    path = Path("data/processed/oos_ens.parquet")
    if path.exists() and "--refresh" not in sys.argv:
        oos = pd.read_parquet(path)
    else:
        oos = oos_predictions(df)
        path.parent.mkdir(parents=True, exist_ok=True)
        oos.to_parquet(path)
    table, per_race = evaluate_simulator(oos, df, raw[["race_id", "driver_id", "classified"]], FIRST_TEST)
    n = per_race["race_id"].nunique()
    print(f"== Calidad de la distribución de posiciones: {n} carreras de test ({FIRST_TEST}-{per_race['season'].max()}) ==")
    print("log_score = media de log P(posición real) (más alto es mejor); cobertura50 y cobertura80 deberían rondar 0,50 y 0,80\n")
    print(table.round(4).to_string())
    s = per_race[per_race["modelo"].isin(["simulador PL", "simulador ajustado a P(win)", "grid empírica"])] \
        .groupby(["season", "modelo"])["log_score"].mean()
    print("\n== log_score por temporada ==")
    print(s.unstack().round(4).to_string())
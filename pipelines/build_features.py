"""CLEAN -> FEATURES. Uso: python -m pipelines.build_features data/f1.duckdb"""
from __future__ import annotations
import sys
from pathlib import Path
from features.build import FEATURES, build_features, load_raw
from storage.db import connect

if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
    df = build_features(load_raw(con))
    con.register("_feat", df)
    con.execute("create or replace table features.driver_race as select * from _feat")
    con.unregister("_feat")
    Path("data/features").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/features/driver_race.parquet")
    print(f"features.driver_race: {df.shape[0]} filas, {len(FEATURES)} features\n")
    print("% de nulos por feature:")
    print((df[FEATURES].isna().mean() * 100).round(1).to_string())
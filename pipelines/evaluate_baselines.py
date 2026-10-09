"""Evalúa los baselines con validación temporal expansiva: para cada temporada T se ajusta con temporadas < T.
Uso: python -m pipelines.evaluate_baselines data/f1.duckdb"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from features.baselines import SlotBaseline, Uniform, metrics
from features.build import build_features, load_raw
from features.temporal import assert_temporal_split
from storage.db import connect

MIN_TRAIN_SEASONS = 3
# nombre -> (constructor, columna necesaria)
MODELS = {"0_azar": (lambda: Uniform(), None), "1_grid": (lambda: SlotBaseline("grid_eff"), "grid_eff"),
          "2_standings": (lambda: SlotBaseline("standings_slot"), "standings_slot"),
          "3_quali": (lambda: SlotBaseline("quali_slot"), "quali_slot"),
          "4_forma": (lambda: SlotBaseline("prior_driver_form_rank"), "prior_driver_form_rank")}


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "field" not in df:            # la tabla de features usa otros nombres para estas columnas
        df["field"] = df["grid_field_size"]
    if "standings_slot" not in df:
        df["standings_slot"] = df["prior_standings_rank"]
    if "finish_position" not in df:  # etiqueta real; se usa solo para evaluar, nunca como feature
        df["finish_position"] = df["target_finish"]
    missing = sorted({col for _, col in MODELS.values() if col and col not in df.columns})
    if missing:
        raise ValueError(f"Faltan columnas para evaluar los baselines: {missing}")
    seasons = sorted(df["season"].unique())
    models = {n: make for n, (make, _) in MODELS.items()}
    rows, pooled = [], {name: [] for name in models}
    for t in seasons[MIN_TRAIN_SEASONS:]:
        tr, te = df[df["season"] < t], df[df["season"] == t].reset_index(drop=True)
        assert_temporal_split((tr["season"] * 100 + tr["round"]).tolist(), (te["season"] * 100 + te["round"]).tolist())
        for name, make in models.items():
            pw, pp, ef = make().fit(tr).predict(te)
            rows.append({"season": t, "baseline": name, **metrics(te, pw, pp, ef)})
            pooled[name].append(te.assign(pw=pw, pp=pp, ef=ef))
    for name, parts in pooled.items():
        all_te = pd.concat(parts, ignore_index=True)
        rows.append({"season": "todas", "baseline": name, **metrics(all_te, all_te["pw"], all_te["pp"], all_te["ef"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = evaluate(build_features(load_raw(connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb"))))
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    res.to_csv("data/processed/baselines.csv", index=False)
    print(res.round(4).to_string(index=False))
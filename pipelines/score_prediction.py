"""Compara una predicción guardada (data/predictions/) con el resultado real y con los baselines.
Uso: python -m pipelines.score_prediction 2026_17 [--file data/predictions/2026_17_....csv] [--db data/f1.duckdb]
Una sola carrera es una anécdota: sirve para acumular una evaluación honesta carrera a carrera."""
from __future__ import annotations
import argparse
import glob
import sys
import numpy as np
import pandas as pd
from features.baselines import SlotBaseline
from features.build import build_features, load_raw
from pipelines.train_eval import _prepare
from storage.db import connect


def _logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(models: dict, real: pd.DataFrame) -> pd.DataFrame:
    """models: nombre -> DataFrame(driver_id, p_win, p_podium, exp_finish). real: driver_id, finish_position.
    Solo se comparan los pilotos presentes en todos los modelos y en el resultado."""
    common = set(real["driver_id"]).intersection(*[set(m["driver_id"]) for m in models.values()])
    rows = []
    for name, m in models.items():
        d = real[real["driver_id"].isin(common)].merge(m, on="driver_id")
        yw, yp = (d["finish_position"] == 1).to_numpy(float), (d["finish_position"] <= 3).to_numpy(float)
        rows.append({"modelo": name, "pilotos": len(d), "P(ganador real)": float(d.loc[yw == 1, "p_win"].iloc[0]),
                     "logloss_win": _logloss(yw, d["p_win"]), "brier_win": float(np.mean((d["p_win"] - yw) ** 2)),
                     "brier_podio": float(np.mean((d["p_podium"] - yp) ** 2)),
                     "MAE_posicion": float(np.mean(np.abs(d["exp_finish"] - d["finish_position"])))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("race_id")
    ap.add_argument("--file")
    ap.add_argument("--db", default="data/f1.duckdb")
    a = ap.parse_args()
    files = sorted(glob.glob(f"data/predictions/{a.race_id}_*.csv"))
    path = a.file or (files[0] if files else None)
    if not path:
        sys.exit(f"No hay predicciones guardadas para {a.race_id} en data/predictions/.")
    con = connect(a.db)
    df = _prepare(build_features(load_raw(con)))
    te = df[df["race_id"] == a.race_id].reset_index(drop=True)
    if te.empty:
        sys.exit(f"{a.race_id} no tiene resultados en la base todavía. Actualiza con ingest_results --force.")
    key = te["season"].iloc[0] * 100 + te["round"].iloc[0]
    tr = df[(df["season"] * 100 + df["round"]) < key]
    pred = pd.read_csv(path)
    models = {f"ens guardada ({pred['created_utc'].iloc[0]})": pred[["driver_id", "p_win", "p_podium", "exp_finish"]]}
    for name, col in {"baseline parrilla real": "grid_eff", "baseline clasificación": "quali_slot"}.items():
        pw, pp, ef = SlotBaseline(col).fit(tr).predict(te)
        models[name] = pd.DataFrame({"driver_id": te["driver_id"], "p_win": pw, "p_podium": pp, "exp_finish": ef})
    real = te[["driver_id", "finish_position"]]
    print(f"\n{a.race_id}: predicción guardada en {path}\n")
    print(score(models, real).round(4).to_string(index=False))
    names = con.execute("select driver_id, name from clean.drivers").df()
    det = pred.merge(real, on="driver_id").merge(names, on="driver_id", how="left").sort_values("finish_position")
    det[["p_win", "p_podium"]] = (100 * det[["p_win", "p_podium"]]).round(1)
    print("\nDetalle (ordenado por resultado real; probabilidades en %):\n")
    print(det[["name", "grid_eff", "p_win", "p_podium", "exp_finish", "finish_position"]]
          .rename(columns={"name": "piloto", "grid_eff": "parrilla", "finish_position": "real"}).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
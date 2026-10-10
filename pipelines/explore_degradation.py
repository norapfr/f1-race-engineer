"""Primera mirada a la degradación de neumáticos con las vueltas de FastF1 (solo descriptiva).
Uso: python -m pipelines.explore_degradation data/f1.duckdb"""
from __future__ import annotations
import sys
import pandas as pd
from features.degradation import clean_dry_laps, stint_slopes
from storage.db import connect

if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
    laps = con.execute("""select race_id, driver_id, lap_number, lap_time_s, compound, tyre_life, stint, is_accurate,
                          pit_in, pit_out, track_status from clean.laps""").df()
    ok = clean_dry_laps(laps)
    s = stint_slopes(ok)
    print(f"Vueltas limpias en seco: {len(ok)}; stints medibles: {len(s)} en {s['race_id'].nunique()} carreras\n")
    q = lambda p: (lambda x: x.quantile(p))
    by = s.groupby("compound").agg(stints=("n", "size"), laps_por_stint=("n", "median"),
                                   bruta_mediana=("slope_raw", "median"), relativa_mediana=("slope_rel", "median"),
                                   rel_p25=("slope_rel", q(0.25)), rel_p75=("slope_rel", q(0.75)))
    by["pct_rel_positiva"] = s.groupby("compound")["slope_rel"].apply(lambda x: (x > 0).mean() * 100)
    print("Pendiente por stint, en segundos por vuelta (positiva = el neumático pierde tiempo al envejecer):")
    print(by.round(4).to_string())
    s["season"] = s["race_id"].str[:4]
    print("\nMediana de la pendiente bruta por temporada y compuesto (s/vuelta):")
    print(s.pivot_table(index="season", columns="compound", values="slope_raw", aggfunc="median").round(4).to_string())
    ok["edad"] = pd.cut(ok["tyre_life"], [0, 5, 10, 15, 20, 25, 30, 40, 100])
    print("\nTiempo relativo al campo (s) por tramo de edad del neumático (descriptivo, mezcla circuitos y pilotos):")
    print(ok.pivot_table(index="edad", columns="compound", values="rel_s", aggfunc="mean", observed=True).round(3).to_string())
"""Ajusta la degradación de neumáticos por era reglamentaria. Uso: python -m pipelines.fit_degradation data/f1.duckdb"""
from __future__ import annotations
import sys
from features.degradation import clean_dry_laps
from models.tyre_degradation import curve_tables, fit_degradation
from storage.db import connect

ERAS = {"2019-2021": (2019, 2021), "2022-2025": (2022, 2025), "2026": (2026, 2026)}

if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
    laps = con.execute("""select race_id, driver_id, lap_number, lap_time_s, compound, tyre_life, stint, is_accurate,
                          pit_in, pit_out, track_status from clean.laps""").df()
    ok = clean_dry_laps(laps)
    season = ok["race_id"].str[:4].astype(int)
    for era, (a, b) in ERAS.items():
        sub = ok[(season >= a) & (season <= b)].reset_index(drop=True)
        fit = fit_degradation(sub)
        table, fresh = curve_tables(fit)
        print(f"\n=== Era {era}: {fit.n_laps} vueltas, {fit.n_races} carreras ===")
        print("Tiempo que pierde el neumático al envejecer, frente a uno de 1 vuelta (s ± error estándar), por edad en vueltas:")
        print(table.to_string())
        print("Nivel de un neumático nuevo frente al medio nuevo (s; negativo = más rápido): " +
              "; ".join(f"{c} {e:+.2f}±{s:.2f}" for c, (e, s) in fresh.items() if c != "MEDIUM"))
"""Predice una carrera ya disputada con el modelo de referencia, usando solo carreras anteriores.
Uso: python -m pipelines.predict_race 2026_17 [data/f1.duckdb] [--sim]
--sim: añade la distribución de posiciones (10.000 carreras simuladas) y la compara con el resultado real."""
from __future__ import annotations
import argparse
import pandas as pd
from features.build import build_features, load_raw
from models.predict import REFERENCE_MODEL, predict_race
from storage.db import connect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("race_id")
    ap.add_argument("db", nargs="?", default="data/f1.duckdb")
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    con = connect(a.db)
    raw = load_raw(con)
    df = build_features(raw)
    pred = predict_race(df, a.race_id)
    names = con.execute("select driver_id, name from clean.drivers").df()
    teams = con.execute("select team_id, name as equipo from clean.teams").df()
    real = df.loc[df["race_id"] == a.race_id, ["driver_id", "target_finish"]].rename(columns={"target_finish": "real"})
    out = pred.merge(names, on="driver_id", how="left").merge(teams, on="team_id", how="left").merge(real, on="driver_id", how="left")
    for c in ("p_win", "p_podium", "p_top5", "p_top10"):
        out[c] = (100 * out[c]).round(1)
    out["exp_finish"] = out["exp_finish"].round(1)
    title = con.execute("select name from clean.races where race_id = ?", [a.race_id]).fetchone()
    print(f"\n{a.race_id} — {title[0] if title else ''} (modelo: {REFERENCE_MODEL}; probabilidades en %)\n")
    print(out[["name", "equipo", "grid_eff", "p_win", "p_podium", "p_top5", "p_top10", "exp_finish", "real"]]
          .rename(columns={"name": "piloto", "grid_eff": "parrilla"}).to_string(index=False))
    print(f"\nSuma P(win) = {pred['p_win'].sum():.2f}; suma P(podio) = {pred['p_podium'].sum():.2f}")
    if a.sim:
        from simulation.live import load_oos, simulate_historical
        sim = simulate_historical(df, raw[["race_id", "driver_id", "classified"]], load_oos(df), pred, a.race_id)
        fin = raw.loc[raw["race_id"] == a.race_id, ["driver_id", "classified"]].rename(columns={"classified": "acabó"})
        t = sim.merge(names, on="driver_id", how="left").merge(real, on="driver_id", how="left").merge(fin, on="driver_id", how="left")
        t["name"], t["acabó"] = t["name"].fillna(t["driver_id"]), t["acabó"].fillna(True).astype(bool)
        t["en_50%"] = ((t["lo50"] <= t["real"]) & (t["real"] <= t["hi50"])).where(t["acabó"])   # solo cuentan quienes acabaron
        t["en_80%"] = ((t["lo80"] <= t["real"]) & (t["real"] <= t["hi80"])).where(t["acabó"])
        t["50%"] = "P" + t["lo50"].astype(str) + "–P" + t["hi50"].astype(str)
        t["80%"] = "P" + t["lo80"].astype(str) + "–P" + t["hi80"].astype(str)
        t[["p_win", "p_podium", "p_dnf"]] = (100 * t[["p_win", "p_podium", "p_dnf"]]).round(1)
        t["exp_pos"] = t["exp_pos"].round(1)
        print("\nSimulación de 10.000 carreras (intervalos de la posición final SI el piloto acaba; P(abandono) aparte):\n")
        print(t[["name", "exp_pos", "50%", "80%", "p_win", "p_podium", "p_dnf", "real", "acabó", "en_50%", "en_80%"]]
              .rename(columns={"name": "piloto", "exp_pos": "pos_esperada", "p_dnf": "P(abandono)",
                               "50%": "50% si acaba", "80%": "80% si acaba"}).to_string(index=False))
        ok = t[t["acabó"]]
        print(f"\nDe los {len(ok)} que acabaron, el real cae dentro del intervalo del 50 %: {ok['en_50%'].astype(float).mean():.0%}; "
              f"del 80 %: {ok['en_80%'].astype(float).mean():.0%}. Abandonos reales: {int((~t['acabó']).sum())} "
              f"(esperados: {t['p_dnf'].sum() / 100:.1f})")
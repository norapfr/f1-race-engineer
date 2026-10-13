"""Predice una carrera FUTURA a partir de su clasificación ya disputada.
Uso: python -m pipelines.predict_upcoming 17 [--season 2026] [--grid-file parrilla.csv] [--sim] [--why] [--save]
Sin conexión con la API: python -m pipelines.predict_upcoming 17 --quali-file quali.csv --circuit marina_bay
(plantilla con: python -m pipelines.quali_template). Antes, actualiza la base con las últimas carreras (ver README)."""
from __future__ import annotations
import argparse
import sys
import pandas as pd
from features.build import build_features, load_raw
from features.upcoming import NoQualifying, load_grid_override, upcoming_rows, upcoming_rows_from_table
from models.predict import REFERENCE_MODEL, predict_race
from storage.db import connect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--db", default="data/f1.duckdb")
    ap.add_argument("--grid-file", help="CSV driver_id,grid con la parrilla real (penalizaciones, pit lane = 0)")
    ap.add_argument("--quali-file", help="CSV de clasificación escrito a mano (driver_id,position[,q1]); no usa la API")
    ap.add_argument("--circuit", help="circuit_id de la carrera (obligatorio con --quali-file), p. ej. marina_bay")
    ap.add_argument("--sim", action="store_true", help="simula 10.000 carreras: distribución de posiciones e intervalos")
    ap.add_argument("--why", action="store_true", help="explica los factores de los 6 favoritos")
    ap.add_argument("--save", action="store_true", help="guarda la predicción en data/predictions/ para evaluarla después")
    a = ap.parse_args()
    con = connect(a.db)
    season = a.season or con.execute("select max(season) from clean.races").fetchone()[0]
    last = con.execute('select coalesce(max("round"), 0) from clean.races where season = ?', [season]).fetchone()[0]
    if last < a.round - 1:
        sys.exit(f"Faltan carreras anteriores en la base (la última es la ronda {last}). Actualiza con:\n"
                 f"  python -m pipelines.ingest_results --from {season} --to {season} --force\n"
                 f"  python -m pipelines.ingest_fastf1 --from {season} --to {season}\n"
                 f"  python -m pipelines.build_pace {a.db}")
    raw = load_raw(con)
    override = load_grid_override(a.grid_file) if a.grid_file else None
    if a.quali_file:
        if not a.circuit:
            sys.exit("Con --quali-file indica el circuito con --circuit (p. ej. marina_bay).")
        last_team = raw.sort_values(["season", "round"]).groupby("driver_id")["team_id"].last().to_dict()
        try:
            up = upcoming_rows_from_table(season, a.round, a.circuit, pd.read_csv(a.quali_file), last_team, override)
        except NoQualifying:
            sys.exit("El archivo no tiene posiciones rellenadas en la columna 'position'.")
        except ValueError as e:
            sys.exit(str(e))
        race_name = f"ronda {a.round} ({a.circuit})"
    else:
        from ingestion.jolpica import JolpicaClient, merge_race_pages
        client = JolpicaClient()
        info = merge_race_pages(client.paged(f"{season}/{a.round}"))
        quali = merge_race_pages(client.paged(f"{season}/{a.round}/qualifying"), "QualifyingResults")
        try:
            up = upcoming_rows(info, quali, override)
        except NoQualifying:
            sys.exit("La clasificación de esta carrera todavía no está disponible. Vuelve a ejecutarlo cuando termine.")
        race_name = info["MRData"]["RaceTable"]["Races"][0]["raceName"]
    race_id = up["race_id"].iloc[0]
    if up["circuit_id"].iloc[0] not in set(raw["circuit_id"]):
        print(f"Aviso: no hay carreras previas en el circuito '{up['circuit_id'].iloc[0]}' en la base; "
              "si no es un circuito nuevo, revisa el nombre (--circuit).")
    if (raw["race_id"] == race_id).any():
        sys.exit(f"{race_id} ya tiene resultados en la base: usa pipelines.predict_race.")
    df = build_features(pd.concat([raw, up], ignore_index=True))
    pred = predict_race(df, race_id)
    sim = None
    if a.sim:
        from simulation.live import RaceSimulator, load_oos
        hist = df[df["race_id"].isin(raw["race_id"])]
        classified = raw[["race_id", "driver_id", "classified"]]
        simulator = RaceSimulator.fit(load_oos(hist), hist, classified)
        sim = simulator.run(pred, df[df["race_id"] == race_id])
    names = con.execute("select driver_id, name from clean.drivers").df()
    teams = con.execute("select team_id, name as equipo from clean.teams").df()
    out = pred.merge(names, on="driver_id", how="left").merge(teams, on="team_id", how="left")
    out["name"], out["equipo"] = out["name"].fillna(out["driver_id"]), out["equipo"].fillna(out["team_id"])
    for c in ("p_win", "p_podium", "p_top5", "p_top10"):
        out[c] = (100 * out[c]).round(1)
    out["exp_finish"] = out["exp_finish"].round(1)
    src = "archivo de parrilla" if override else "clasificación, sin penalizaciones"
    print(f"\n{race_id} — {race_name} (modelo: {REFERENCE_MODEL}; probabilidades en %; parrilla = {src})\n")
    print(out[["name", "equipo", "grid_eff", "p_win", "p_podium", "p_top5", "p_top10", "exp_finish"]]
          .rename(columns={"name": "piloto", "grid_eff": "parrilla"}).to_string(index=False))
    print(f"\nSuma P(win) = {pred['p_win'].sum():.2f}; suma P(podio) = {pred['p_podium'].sum():.2f}")
    if sim is not None:
        t = sim.merge(names, on="driver_id", how="left")
        t["name"] = t["name"].fillna(t["driver_id"])
        t["50%"] = "P" + t["lo50"].astype(str) + "–P" + t["hi50"].astype(str)
        t["80%"] = "P" + t["lo80"].astype(str) + "–P" + t["hi80"].astype(str)
        t[["p_win", "p_podium", "p_dnf"]] = (100 * t[["p_win", "p_podium", "p_dnf"]]).round(1)
        t["exp_pos"] = t["exp_pos"].round(1)
        print("\nSimulación de 10.000 carreras (intervalos de la posición final SI el piloto acaba; P(abandono) aparte):\n")
        print(t[["name", "exp_pos", "50%", "80%", "p_win", "p_podium", "p_dnf"]]
              .rename(columns={"name": "piloto", "exp_pos": "pos_esperada", "p_dnf": "P(abandono)",
                               "50%": "50% si acaba", "80%": "80% si acaba"}).to_string(index=False))
    if a.why:
        from models.explain import explain_groups, explain_race
        pd.set_option("display.width", 250)
        table = explain_groups(explain_race(df, race_id), out.set_index("driver_id")["name"], list(out["driver_id"].head(6)))
        print("\nPor qué: efecto de cada grupo de factores sobre las probabilidades de ganar "
              "(x = multiplica las probabilidades frente a un piloto medio):\n")
        print(table.round(2).to_string(index=False))
    if a.save:
        from datetime import datetime, timezone
        from pathlib import Path
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path("data/predictions") / f"{race_id}_{stamp}.csv"
        saved = pred if sim is None else pred.merge(sim.drop(columns=["p_win", "p_podium", "p_top5", "p_top10"])
                                                    .add_prefix("sim_").rename(columns={"sim_driver_id": "driver_id"}),
                                                    on="driver_id", how="left")
        saved.assign(model=REFERENCE_MODEL, created_utc=stamp).to_csv(path, index=False)
        print(f"\nPredicción guardada en {path}")


if __name__ == "__main__":
    main()
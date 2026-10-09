"""Predice una carrera FUTURA a partir de su clasificación ya disputada.
Uso: python -m pipelines.predict_upcoming 17 [--season 2026] [--grid-file parrilla.csv] [--why] [--db data/f1.duckdb]
Antes, actualiza la base con las últimas carreras (ver README)."""
from __future__ import annotations
import argparse
import sys
import pandas as pd
from features.build import build_features, load_raw
from features.upcoming import NoQualifying, load_grid_override, upcoming_rows
from ingestion.jolpica import JolpicaClient, merge_race_pages
from models.predict import REFERENCE_MODEL, predict_race
from storage.db import connect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--db", default="data/f1.duckdb")
    ap.add_argument("--grid-file", help="CSV driver_id,grid con la parrilla real (penalizaciones, pit lane = 0)")
    ap.add_argument("--why", action="store_true", help="explica los factores de los 6 favoritos")
    a = ap.parse_args()
    con = connect(a.db)
    season = a.season or con.execute("select max(season) from clean.races").fetchone()[0]
    last = con.execute('select coalesce(max("round"), 0) from clean.races where season = ?', [season]).fetchone()[0]
    if last < a.round - 1:
        sys.exit(f"Faltan carreras anteriores en la base (la última es la ronda {last}). Actualiza con:\n"
                 f"  python -m pipelines.ingest_results --from {season} --to {season} --force\n"
                 f"  python -m pipelines.ingest_fastf1 --from {season} --to {season}\n"
                 f"  python -m pipelines.build_pace {a.db}")
    client = JolpicaClient()
    info = merge_race_pages(client.paged(f"{season}/{a.round}"))
    quali = merge_race_pages(client.paged(f"{season}/{a.round}/qualifying"), "QualifyingResults")
    override = load_grid_override(a.grid_file) if a.grid_file else None
    try:
        up = upcoming_rows(info, quali, override)
    except NoQualifying:
        sys.exit("La clasificación de esta carrera todavía no está disponible. Vuelve a ejecutarlo cuando termine.")
    race_id = up["race_id"].iloc[0]
    raw = load_raw(con)
    if (raw["race_id"] == race_id).any():
        sys.exit(f"{race_id} ya tiene resultados en la base: usa pipelines.predict_race.")
    df = build_features(pd.concat([raw, up], ignore_index=True))
    pred = predict_race(df, race_id)
    names = con.execute("select driver_id, name from clean.drivers").df()
    teams = con.execute("select team_id, name as equipo from clean.teams").df()
    out = pred.merge(names, on="driver_id", how="left").merge(teams, on="team_id", how="left")
    out["name"], out["equipo"] = out["name"].fillna(out["driver_id"]), out["equipo"].fillna(out["team_id"])
    for c in ("p_win", "p_podium", "p_top5", "p_top10"):
        out[c] = (100 * out[c]).round(1)
    out["exp_finish"] = out["exp_finish"].round(1)
    race_name = info["MRData"]["RaceTable"]["Races"][0]["raceName"]
    print(f"\n{race_id} — {race_name} (modelo: {REFERENCE_MODEL}; probabilidades en %; parrilla = "
          f"{'archivo' if override else 'clasificación, sin penalizaciones'})\n")
    print(out[["name", "equipo", "grid_eff", "p_win", "p_podium", "p_top5", "p_top10", "exp_finish"]]
          .rename(columns={"name": "piloto", "grid_eff": "parrilla"}).to_string(index=False))
    print(f"\nSuma P(win) = {pred['p_win'].sum():.2f}; suma P(podio) = {pred['p_podium'].sum():.2f}")
    if a.why:
        from models.explain import explain_groups, explain_race
        pd.set_option("display.width", 250)
        table = explain_groups(explain_race(df, race_id), out.set_index("driver_id")["name"], list(out["driver_id"].head(6)))
        print("\nPor qué: efecto de cada grupo de factores sobre las probabilidades de ganar "
                "(x = multiplica las probabilidades frente a un piloto medio):\n")
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
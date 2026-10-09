"""FastF1 (carreras) -> RAW -> CLEAN. Reanudable: lo ya descargado no se vuelve a bajar.
Uso:  python -m pipelines.ingest_fastf1 --from 2023 --to 2023 --limit 2    (prueba corta)
      python -m pipelines.ingest_fastf1 --from 2018 --to 2018              (una temporada)"""
from __future__ import annotations
import argparse
import json
import pandas as pd
from ingestion.fastf1_source import fetch_session_raw
from ingestion.normalize_fastf1 import normalize_laps, normalize_race_control, normalize_weather
from ingestion.raw_store import RawStore
from storage.db import connect, upsert


def code_map(con) -> tuple[dict, set]:
    rows = con.execute("select code, driver_id from clean.drivers where code is not null").fetchall()
    seen, ambiguous = {}, set()
    for code, did in rows:
        if code in seen and seen[code] != did:
            ambiguous.add(code)
        seen[code] = did
    return {c: d for c, d in seen.items() if c not in ambiguous}, ambiguous


def load_race(con, store: RawStore, race_id: str, key: str, cmap: dict) -> dict:
    rd = lambda ds: pd.read_parquet(store.path("fastf1", ds, key, "parquet"))
    laps, unmatched = normalize_laps(rd("laps"), race_id, cmap)
    for table, df, keys in [("laps", laps, ["race_id", "driver_id", "lap_number"]),
                            ("weather", normalize_weather(rd("weather"), race_id), None),
                            ("race_control", normalize_race_control(rd("race_control"), race_id), None)]:
        if keys is None:   # sin clave natural: se reemplaza por carrera
            con.execute(f"delete from clean.{table} where race_id = ?", [race_id])
            if not df.empty:
                con.register("_t", df)
                con.execute(f"insert into clean.{table} by name select * from _t")
                con.unregister("_t")
        else:
            upsert(con, f"clean.{table}", df, keys)
    return {"laps": len(laps), "sin_correspondencia": sorted(unmatched)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, required=True)
    ap.add_argument("--to", dest="y1", type=int, required=True)
    ap.add_argument("--db", default="data/f1.duckdb")
    ap.add_argument("--limit", type=int, default=None, help="máximo de carreras (para pruebas)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    store, con = RawStore(), connect(a.db)
    cmap, ambiguous = code_map(con)
    if ambiguous:
        print("Aviso: códigos de piloto ambiguos (se ignoran):", sorted(ambiguous))
    races = con.execute('select race_id, season, "round" from clean.races where season between ? and ? order by 1',
                        [a.y0, a.y1]).fetchall()[: a.limit]
    failures = []
    for race_id, season, rnd in races:
        key = f"{season}_{rnd:02d}_R"
        try:
            fetch_session_raw(season, rnd, "R", store, force=a.force)
            print(race_id, load_race(con, store, race_id, key, cmap), flush=True)
        except Exception as e:  # una carrera rota no debe parar la descarga
            failures.append({"race_id": race_id, "error": repr(e)})
            print(race_id, "FALLO:", repr(e)[:200], flush=True)
    if failures:
        with open(store.root / "fastf1_failures.jsonl", "a") as f:
            f.writelines(json.dumps(x) + "\n" for x in failures)
    print(f"\nHecho: {len(races) - len(failures)} carreras OK, {len(failures)} con fallo.")


if __name__ == "__main__":
    main()
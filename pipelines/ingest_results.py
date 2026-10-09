"""RAW(Jolpica) -> CLEAN(DuckDB). Uso: python -m pipelines.ingest_results --from 2018 --to 2026 --db data/f1.duckdb"""
from __future__ import annotations
import argparse
from ingestion.jolpica import JolpicaClient
from ingestion.normalize import normalize_qualifying, normalize_results
from ingestion.raw_store import RawStore
from storage.db import connect, upsert

KEYS = {"circuits": ["circuit_id"], "races": ["race_id"], "drivers": ["driver_id"],
        "teams": ["team_id"], "race_results": ["race_id", "driver_id"]}


def load_payload(con, payload: dict) -> dict[str, int]:
    tables = normalize_results(payload)
    return {t: upsert(con, f"clean.{t}", df, KEYS[t]) for t, df in tables.items()}


def load_qualifying(con, payload: dict) -> int:
    return upsert(con, "clean.qualifying_results", normalize_qualifying(payload), ["race_id", "driver_id"])


def get_or_fetch(store, dataset, season, fetch, force):
    key = str(season)
    if store.exists("jolpica", dataset, key, "json") and not force:
        return store.read_json("jolpica", dataset, key)
    payload = fetch(season)
    store.write_json("jolpica", dataset, key, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, required=True)
    ap.add_argument("--to", dest="y1", type=int, required=True)
    ap.add_argument("--db", default="data/f1.duckdb")
    ap.add_argument("--force", action="store_true", help="vuelve a descargar aunque exista en raw")
    a = ap.parse_args()
    store, client, con = RawStore(), JolpicaClient(), connect(a.db)
    for season in range(a.y0, a.y1 + 1):
        results = get_or_fetch(store, "results", season, client.season_results, a.force)
        quali = get_or_fetch(store, "qualifying", season, client.season_qualifying, a.force)
        print(season, load_payload(con, results), {"qualifying": load_qualifying(con, quali)})


if __name__ == "__main__":
    main()
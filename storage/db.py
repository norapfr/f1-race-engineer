"""Acceso a DuckDB: creación de esquema y upserts idempotentes."""
from __future__ import annotations
from pathlib import Path
import duckdb
import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute(SCHEMA_PATH.read_text())
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, keys: list[str]) -> int:
    """Borra las filas con la misma clave e inserta las nuevas (por nombre de columna)."""
    if df.empty:
        return 0
    df = df.drop_duplicates(subset=keys, keep="last")
    con.register("_incoming", df)
    try:
        cond = " AND ".join(f"t.{k} = d.{k}" for k in keys)
        con.execute(f"DELETE FROM {table} t USING _incoming d WHERE {cond}")
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    return len(df)

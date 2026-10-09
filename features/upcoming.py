"""Filas de una carrera FUTURA (sin resultados) a partir de su calendario y su clasificación ya disputada.
Los campos de resultado van vacíos: ninguna feature los usa para esa misma carrera (solo de las anteriores)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ingestion.normalize import normalize_qualifying, normalize_results


class NoQualifying(Exception):
    """La clasificación de la carrera aún no está disponible."""


def load_grid_override(path: str) -> dict:
    """CSV con columnas driver_id,grid (grid 0 = salida desde el pit lane)."""
    d = pd.read_csv(path)
    return dict(zip(d["driver_id"], d["grid"].astype(int)))


def upcoming_rows(info: dict, quali: dict, grid_override: dict | None = None) -> pd.DataFrame:
    race = normalize_results(info)["races"]
    if race.empty:
        raise ValueError("La fuente no tiene esa carrera en el calendario.")
    r = race.iloc[0]
    q = normalize_qualifying(quali)
    if q.empty:
        raise NoQualifying(r["race_id"])
    grid = q["position"].astype(float)
    if grid_override:
        grid = q["driver_id"].map(grid_override).fillna(grid)
    return pd.DataFrame({
        "race_id": r["race_id"], "season": int(r["season"]), "round": int(r["round"]), "circuit_id": r["circuit_id"],
        "driver_id": q["driver_id"].to_numpy(), "team_id": q["team_id"].to_numpy(), "grid": grid.to_numpy(),
        # sin resultados: points=0 y classified=True son neutros (el resultado propio nunca entra en sus features)
        "finish_position": np.nan, "points": 0.0, "classified": True,
        "quali_pos": q["position"].astype(float).to_numpy(), "q1_s": q["q1_s"].to_numpy(), "race_pace_pct": np.nan})
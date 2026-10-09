"""Registro de disponibilidad temporal de cada columna. Política: FAIL-CLOSED.
Una feature solo es válida si está registrada y su disponibilidad <= el momento de corte de la predicción."""
from __future__ import annotations
from enum import IntEnum


class Avail(IntEnum):
    PRE_WEEKEND = 0  # conocido antes del fin de semana (histórico previo, metadatos de circuito)
    POST_QUALI = 1   # conocido con la parrilla definida (grid, tiempos de Q, previsión meteo actualizada)
    POST_RACE = 2    # solo existe tras la bandera a cuadros: NUNCA puede ser feature


class LeakageError(ValueError):
    pass


IDENTIFIERS = {"race_id", "driver_id", "team_id", "circuit_id", "season", "round"}

EXPLICIT = {
    "finish_position": Avail.POST_RACE, "points": Avail.POST_RACE, "status": Avail.POST_RACE,
    "laps_completed": Avail.POST_RACE, "classified": Avail.POST_RACE, "position_text": Avail.POST_RACE,
    "grid": Avail.POST_QUALI,
}
PREFIX_RULES = (
    ("target_", Avail.POST_RACE), ("race_", Avail.POST_RACE), ("actual_", Avail.POST_RACE),
    ("grid_", Avail.POST_QUALI), ("quali_", Avail.POST_QUALI), ("forecast_", Avail.POST_QUALI),
    ("prior_", Avail.PRE_WEEKEND),   # solo válido si se construye con features.temporal (shift previo)
    ("circuit_", Avail.PRE_WEEKEND),
)


def classify(col: str) -> Avail:
    if col in EXPLICIT:
        return EXPLICIT[col]
    for prefix, avail in PREFIX_RULES:
        if col.startswith(prefix):
            return avail
    raise LeakageError(f"Columna sin registrar: '{col}'. Regístrala en features/availability.py antes de usarla.")


def check_features(cols, cutoff: Avail = Avail.POST_QUALI) -> None:
    """Lanza LeakageError si alguna feature no está permitida en el momento de corte."""
    bad = {c: classify(c).name for c in cols if c not in IDENTIFIERS and classify(c) > cutoff}
    if bad:
        raise LeakageError(f"Features no disponibles antes de la carrera: {bad}")

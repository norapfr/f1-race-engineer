"""Ingesta FastF1: SOLO se usa en pipelines offline, nunca en tiempo de ejecución de la app.
Requiere `pip install fastf1` y acceso de red a los servidores de F1."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .raw_store import RawStore

# columnas de vueltas que necesitamos (el resto se descarta del RAW para que sea ligero y serializable)
LAP_COLS = ["Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "Compound", "TyreLife", "Stint", "PitInTime", "PitOutTime", "Position", "TrackStatus", "IsAccurate"]


def _write(store: RawStore, dataset: str, key: str, df: pd.DataFrame) -> None:
    df = pd.DataFrame(df).reset_index(drop=True)
    try:
        store.write_parquet("fastf1", dataset, key, df)
    except Exception:  # columnas object con tipos mezclados: se guardan como texto
        obj = df.select_dtypes("object").columns
        store.write_parquet("fastf1", dataset, key, df.astype({c: str for c in obj}))


def fetch_session_raw(season: int, rnd: int, session: str, store: RawStore,
                      cache_dir: str | Path = "data/fastf1_cache", force: bool = False) -> None:
    """session: 'R' (carrera), 'Q' (clasificación), 'S' (sprint), 'FP1'... Sin telemetría de coche (pesada).
    El último fichero que se escribe es race_control: si existe, la sesión se descargó entera."""
    import fastf1  # import perezoso: el resto del proyecto no depende de FastF1
    key = f"{season}_{rnd:02d}_{session}"
    if store.exists("fastf1", "race_control", key, "parquet") and not force:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    fastf1.set_log_level("WARNING")
    s = fastf1.get_session(season, rnd, session)
    s.load(laps=True, telemetry=False, weather=True, messages=True)
    laps = pd.DataFrame(s.laps)
    _write(store, "results", key, s.results)
    _write(store, "laps", key, laps[[c for c in LAP_COLS if c in laps.columns]])
    weather = s.weather_data if s.weather_data is not None else pd.DataFrame()
    _write(store, "weather", key, weather)
    rc = getattr(s, "race_control_messages", None)
    _write(store, "race_control", key, rc if rc is not None else pd.DataFrame())
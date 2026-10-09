"""RAW FastF1 -> CLEAN. Funciones puras (sin red), tolerantes a columnas ausentes entre versiones de FastF1."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")


def _secs(s: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(s):
        return s.dt.total_seconds()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")          # ya en segundos
    return pd.to_timedelta(s, errors="coerce").dt.total_seconds()


_NULL_TEXT = {"nan", "none", "<na>", ""}


def _text(s: pd.Series) -> pd.Series:
    """Texto con nulos reales: el RAW puede traer 'nan' o 'None' como cadenas."""
    s = s.astype("object")
    return s.where(s.notna() & ~s.astype(str).str.strip().str.lower().isin(_NULL_TEXT), None)


def _int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def normalize_laps(laps: pd.DataFrame, race_id: str, code_to_driver: dict) -> tuple[pd.DataFrame, set]:
    """Devuelve (tabla clean.laps, códigos de piloto sin correspondencia en Jolpica)."""
    codes = laps["Driver"].astype(str)
    out = pd.DataFrame({
        "race_id": race_id, "driver_id": codes.map(code_to_driver), "lap_number": _int(_col(laps, "LapNumber")),
        "lap_time_s": _secs(_col(laps, "LapTime")), "sector1_s": _secs(_col(laps, "Sector1Time")),
        "sector2_s": _secs(_col(laps, "Sector2Time")), "sector3_s": _secs(_col(laps, "Sector3Time")),
        "compound": _text(_col(laps, "Compound")),
        "tyre_life": _int(_col(laps, "TyreLife")), "stint": _int(_col(laps, "Stint")),
        "position": _int(_col(laps, "Position")),
        "pit_in": _col(laps, "PitInTime").notna(), "pit_out": _col(laps, "PitOutTime").notna(),
        "track_status": _text(_col(laps, "TrackStatus")),
        "is_accurate": (_col(laps, "IsAccurate").fillna(False).astype(bool) if "IsAccurate" in laps.columns
                        else _secs(_col(laps, "LapTime")).notna()),
    })
    unmatched = set(codes[out["driver_id"].isna()].unique())
    out = out.dropna(subset=["driver_id", "lap_number"])
    out["lap_number"] = out["lap_number"].astype(int)
    out = out.drop_duplicates(["race_id", "driver_id", "lap_number"], keep="last").reset_index(drop=True)
    return out, unmatched


def normalize_weather(w: pd.DataFrame, race_id: str) -> pd.DataFrame:
    if w is None or w.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "race_id": race_id, "t_offset_s": _secs(_col(w, "Time")), "air_temp": _col(w, "AirTemp"),
        "track_temp": _col(w, "TrackTemp"), "humidity": _col(w, "Humidity"), "pressure": _col(w, "Pressure"),
        "wind_speed": _col(w, "WindSpeed"), "rainfall": _col(w, "Rainfall").fillna(False).astype(bool)})


def normalize_race_control(rc: pd.DataFrame, race_id: str) -> pd.DataFrame:
    if rc is None or rc.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "race_id": race_id, "t_offset_s": np.nan, "lap": _int(_col(rc, "Lap")),
        "category": _col(rc, "Category").astype("object"), "message": _col(rc, "Message").astype("object")})
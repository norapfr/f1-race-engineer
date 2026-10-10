"""Punto de partida del modelo de degradación de neumáticos: vueltas limpias en seco y pendiente por stint.
Dos medidas por stint: tiempo de vuelta bruto vs edad del neumático (mezcla degradación y combustible) y tiempo
relativo a la mediana del campo en esa vuelta (quita lo común a todos, pero también la degradación común)."""
from __future__ import annotations
import pandas as pd
from features.pace import MIN_FIELD, WET, WET_SHARE

SLICKS = ("SOFT", "MEDIUM", "HARD")
KEYS = ["race_id", "driver_id", "stint", "compound"]


def clean_dry_laps(laps: pd.DataFrame) -> pd.DataFrame:
    d = laps.copy()
    for c in ("is_accurate", "pit_in", "pit_out"):
        d[c] = d[c].fillna(False).astype(bool)
    wet = d["compound"].isin(WET).groupby(d["race_id"]).mean() >= WET_SHARE
    ok = d[(d["lap_number"] > 1) & d["lap_time_s"].notna() & d["is_accurate"] & ~d["pit_in"] & ~d["pit_out"]
           & (d["track_status"] == "1") & d["compound"].isin(SLICKS) & d["tyre_life"].notna()
           & ~d["race_id"].map(wet).fillna(False).astype(bool)].copy()
    g = ok.groupby(["race_id", "lap_number"])["lap_time_s"]
    ok["med"], ok["cnt"] = g.transform("median"), g.transform("count")
    ok = ok[ok["cnt"] >= MIN_FIELD].copy()
    ok["rel_s"] = ok["lap_time_s"] - ok["med"]
    return ok[ok["rel_s"].abs() <= 0.07 * ok["med"]].reset_index(drop=True)


def stint_slopes(ok: pd.DataFrame, min_laps: int = 8, min_span: int = 7) -> pd.DataFrame:
    """Pendiente (s/vuelta) por stint, por mínimos cuadrados, del tiempo bruto y del relativo frente a la edad."""
    x = ok["tyre_life"].astype(float)
    d = ok.assign(x=x, x2=x * x, xy_raw=x * ok["lap_time_s"], xy_rel=x * ok["rel_s"])
    s = d.groupby(KEYS).agg(n=("x", "size"), sx=("x", "sum"), sx2=("x2", "sum"), xmin=("x", "min"), xmax=("x", "max"),
                            sy_raw=("lap_time_s", "sum"), sxy_raw=("xy_raw", "sum"),
                            sy_rel=("rel_s", "sum"), sxy_rel=("xy_rel", "sum"))
    den = s["n"] * s["sx2"] - s["sx"] ** 2
    s["slope_raw"] = (s["n"] * s["sxy_raw"] - s["sx"] * s["sy_raw"]) / den
    s["slope_rel"] = (s["n"] * s["sxy_rel"] - s["sx"] * s["sy_rel"]) / den
    s = s[(s["n"] >= min_laps) & ((s["xmax"] - s["xmin"]) >= min_span) & (den > 0)]
    return s.reset_index()[KEYS + ["n", "slope_raw", "slope_rel"]]
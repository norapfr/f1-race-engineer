"""Reconstruye la clasificación de una carrera con el cronometraje de FastF1 (sin pasar por Jolpica) y la guarda como CSV
compatible con `predict_upcoming --quali-file`. Uso: python -m pipelines.quali_from_fastf1 17 [--season 2026] [--out quali.csv]"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd


def _codes(laps) -> set:
    return set() if laps is None or len(laps) == 0 else set(pd.DataFrame(laps)["Driver"])


def best_times(laps) -> pd.Series:
    """Mejor vuelta válida (s) por código de piloto en un tramo: sin tiempo ni vueltas borradas por la dirección de carrera."""
    if laps is None or len(laps) == 0:
        return pd.Series(dtype=float)
    d = pd.DataFrame(laps)
    ok = d["LapTime"].notna()
    if "Deleted" in d.columns:
        ok &= ~d["Deleted"].fillna(False).astype(bool)
    return d.loc[ok].groupby("Driver")["LapTime"].min().dt.total_seconds()


def classification_from_segments(q1, q2, q3, all_codes) -> pd.DataFrame:
    """Orden final: los de Q3 por su tiempo de Q3; luego los eliminados en Q2 por su tiempo de Q2; luego los eliminados en Q1
    por su tiempo de Q1; quien no marcó tiempo va detrás de los de su grupo y quien no salió, al final."""
    b1, b2, b3 = best_times(q1), best_times(q2), best_times(q3)
    in3, in2, in1 = _codes(q3), _codes(q2) - _codes(q3), _codes(q1) - _codes(q2)
    order = lambda grp, best: sorted(grp, key=lambda c: (c not in best.index, best.get(c, np.inf), c))
    ranked = order(in3, b3) + order(in2, b2) + order(in1, b1)
    ranked += sorted(set(all_codes) - set(ranked))
    return pd.DataFrame({"code": ranked, "position": range(1, len(ranked) + 1),
                         "q1": [float(b1.get(c, np.nan)) for c in ranked]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--out", default="quali.csv")
    ap.add_argument("--db", default="data/f1.duckdb")
    a = ap.parse_args()
    from pipelines.ingest_fastf1 import code_map
    from storage.db import connect
    con = connect(a.db)
    season = a.season or con.execute("select max(season) from clean.races").fetchone()[0]
    import fastf1
    fastf1.Cache.enable_cache("data/fastf1_cache")
    fastf1.set_log_level("WARNING")
    s = fastf1.get_session(season, a.round, "Q")
    s.load(laps=True, telemetry=False, weather=False, messages=True)
    try:
        q1, q2, q3 = s.laps.split_qualifying_sessions()
    except Exception as e:
        sys.exit(f"No se pudo separar Q1/Q2/Q3 (¿clasificación aún no disponible?): {e!r}")
    table = classification_from_segments(q1, q2, q3, list(s.laps["Driver"].unique()))
    cmap, _ = code_map(con)
    table["driver_id"] = table["code"].map(cmap)
    if table["driver_id"].isna().any():
        sys.exit(f"Códigos sin correspondencia en la base: {list(table.loc[table['driver_id'].isna(), 'code'])}")
    table[["driver_id", "code", "position", "q1"]].rename(columns={"code": "piloto"}).to_csv(a.out, index=False)
    print(table[["position", "code", "q1"]].to_string(index=False))
    print(f"\nGuardado en {a.out}. Compáralo con la clasificación oficial y ejecuta:\n"
          f"  python -m pipelines.predict_upcoming {a.round} --quali-file {a.out} --circuit <circuit_id> --why")


if __name__ == "__main__":
    main()
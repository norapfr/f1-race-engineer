"""¿Son estables entre temporadas las curvas de degradación? Si las rarezas se repiten año tras año, son estructurales
(p. ej. etiquetas de compuesto relativas a cada carrera); si cambian, es ruido.
Uso: python -m pipelines.degradation_by_season data/f1.duckdb"""
from __future__ import annotations
import sys
import pandas as pd
from features.degradation import clean_dry_laps
from models.tyre_degradation import COMPOUNDS, fit_degradation
from storage.db import connect


def by_season(ok: pd.DataFrame, first: int = 2022) -> dict:
    """Devuelve tablas (temporada x compuesto) con 'estimación±error' de: pérdida a 10 y a 20 vueltas, y nivel nuevo vs medio."""
    season = ok["race_id"].str[:4].astype(int)
    rows = []
    for s in sorted(season[season >= first].unique()):
        fit = fit_degradation(ok[season == s].reset_index(drop=True))
        for c in COMPOUNDS:
            (l10, e10), (l20, e20), (off, eo) = fit.loss(c, 10), fit.loss(c, 20), fit.fresh_offset(c)
            rows.append(dict(season=s, compuesto=c, carreras=fit.n_races, perdida_10=f"{l10:+.2f}±{e10:.2f}",
                             perdida_20=f"{l20:+.2f}±{e20:.2f}", nuevo_vs_medio=f"{off:+.2f}±{eo:.2f}"))
    d = pd.DataFrame(rows)
    return {k: d.pivot(index="season", columns="compuesto", values=k)[list(COMPOUNDS)]
            for k in ("perdida_10", "perdida_20", "nuevo_vs_medio")} | {"carreras": d.groupby("season")["carreras"].first()}


if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
    laps = con.execute("""select race_id, driver_id, lap_number, lap_time_s, compound, tyre_life, stint, is_accurate,
                          pit_in, pit_out, track_status from clean.laps""").df()
    t = by_season(clean_dry_laps(laps))
    print("Carreras por temporada:", t["carreras"].to_dict())
    for k, title in [("perdida_10", "Pérdida a las 10 vueltas frente a 1 vuelta (s)"),
                     ("perdida_20", "Pérdida a las 20 vueltas frente a 1 vuelta (s)"),
                     ("nuevo_vs_medio", "Nivel de neumático nuevo frente al medio nuevo (s; negativo = más rápido)")]:
        print(f"\n{title}:")
        print(t[k].to_string())
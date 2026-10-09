"""Detalle de discrepancias entre vueltas de FastF1 y resultados de Jolpica.
Uso: python -m pipelines.check_laps_detail data/f1.duckdb"""
import sys
from storage.db import connect

con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
show = lambda title, sql: print(f"\n{title}\n" + con.execute(sql).df().to_string(index=False))

show("Pilotos con resultado pero sin ninguna vuelta (en carreras con vueltas cargadas):", """
    select x.race_id, x.driver_id, x.grid, x.laps_completed, x.status
    from clean.race_results x
    where exists (select 1 from clean.laps l where l.race_id = x.race_id)
      and not exists (select 1 from clean.laps l where l.race_id = x.race_id and l.driver_id = x.driver_id)
    order by 1, 2""")
show("Desfase de vueltas por carrera (FastF1 - Jolpica), solo carreras con diferencias de más de 1:", """
    with d as (
      select x.race_id, x.driver_id, max(l.lap_number) - x.laps_completed as diff
      from clean.race_results x join clean.laps l on l.race_id = x.race_id and l.driver_id = x.driver_id
      group by x.race_id, x.driver_id, x.laps_completed)
    select race_id, count(*) as pilotos, sum((abs(diff) > 1)::int) as con_dif, median(diff) as dif_mediana,
           min(diff) as dif_min, max(diff) as dif_max
    from d group by race_id having sum((abs(diff) > 1)::int) > 0 order by 1""")
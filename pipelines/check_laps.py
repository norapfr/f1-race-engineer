"""Comprueba que las vueltas de FastF1 cuadran con los resultados de Jolpica. Uso: python -m pipelines.check_laps data/f1.duckdb"""
import sys
from storage.db import connect

con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
show = lambda title, sql: print(f"\n{title}\n" + con.execute(sql).df().to_string(index=False))

show("Carreras con vueltas cargadas por temporada (frente a carreras totales):", """
    select r.season, count(*) as carreras, sum((exists (select 1 from clean.laps l where l.race_id = r.race_id))::int) as con_vueltas
    from clean.races r group by 1 order by 1""")
show("Carreras donde el nº de pilotos con vueltas no coincide con el de resultados:", """
    select r.race_id, count(distinct l.driver_id) as pilotos_vueltas,
           (select count(*) from clean.race_results x where x.race_id = r.race_id) as pilotos_resultados
    from clean.races r join clean.laps l on l.race_id = r.race_id group by r.race_id
    having pilotos_vueltas <> pilotos_resultados order by 1""")
show("Pilotos cuyas vueltas máximas difieren en más de 1 de las vueltas de Jolpica (primeras 25):", """
    select x.race_id, x.driver_id, x.laps_completed as vueltas_jolpica, max(l.lap_number) as vueltas_fastf1
    from clean.race_results x join clean.laps l on l.race_id = x.race_id and l.driver_id = x.driver_id
    group by 1, 2, 3 having abs(x.laps_completed - max(l.lap_number)) > 1 order by 1 limit 25""")
show("Calidad de las vueltas (todas las cargadas):", """
    select count(*) as vueltas, round(100 * avg((lap_time_s is not null)::int), 1) as pct_con_tiempo,
           round(100 * avg(is_accurate::int), 1) as pct_precisas, round(100 * avg((compound is not null)::int), 1) as pct_con_compuesto,
           round(100 * avg((tyre_life is not null)::int), 1) as pct_con_edad_neumatico from clean.laps""")
show("Compuestos:", "select compound, count(*) as vueltas from clean.laps group by 1 order by 2 desc")
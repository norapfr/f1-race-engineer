"""Comprobaciones rápidas de coherencia sobre la capa clean. Uso: python -m pipelines.check_db data/f1.duckdb"""
import sys
from storage.db import connect

con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")

CHECKS = {
    "carreras sin exactamente un ganador": """
        select count(*) from (select race_id from clean.race_results
        where finish_position = 1 group by 1 having count(*) <> 1)""",
    "carreras sin resultados": """
        select count(*) from clean.races r
        where not exists (select 1 from clean.race_results x where x.race_id = r.race_id)""",
    "resultados duplicados (race_id, driver_id)": """
        select count(*) from (select 1 from clean.race_results group by race_id, driver_id having count(*) > 1)""",
    "resultados con equipo o piloto huérfano": """
        select count(*) from clean.race_results x
        where x.driver_id not in (select driver_id from clean.drivers)
           or x.team_id not in (select team_id from clean.teams)""",
}
for name, sql in CHECKS.items():
    n = con.execute(sql).fetchone()[0]
    print(f"{'OK ' if n == 0 else 'MAL'} {name}: {n}")

print("\nGanadores por temporada y número de victorias:")
print(con.execute("""
    select r.season, d.name, count(*) as wins
    from clean.race_results x join clean.races r using (race_id) join clean.drivers d using (driver_id)
    where x.finish_position = 1 group by 1, 2 order by 1, 3 desc""").df().to_string(index=False))

print("\nAbandonos y salidas desde pit lane por temporada:")
print(con.execute("""
    select r.season, sum((not x.classified)::int) as no_clasificados, sum((x.grid = 0)::int) as pit_lane
    from clean.race_results x join clean.races r using (race_id) group by 1 order by 1""").df().to_string(index=False))


print("\nPilotos en la carrera sin fila de clasificación:")
print(con.execute("""
    select x.race_id, d.name, x.grid, x.status
    from clean.race_results x
    join clean.drivers d on d.driver_id = x.driver_id
    left join clean.qualifying_results q on q.race_id = x.race_id and q.driver_id = x.driver_id
    where q.driver_id is null order by 1""").df().to_string(index=False))

print("\nCarreras donde ningún piloto tiene tiempo de clasificación:")
print(con.execute("""
    select race_id, count(*) as filas from clean.qualifying_results
    group by 1
    having sum((q1_s is not null or q2_s is not null or q3_s is not null)::int) = 0""").df().to_string(index=False))
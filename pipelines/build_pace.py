"""clean.laps -> features.race_pace. Uso: python -m pipelines.build_pace data/f1.duckdb"""
from __future__ import annotations
import sys
from features.pace import race_pace_table
from storage.db import connect

if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb")
    laps = con.execute("""select race_id, driver_id, lap_number, lap_time_s, compound, is_accurate, pit_in, pit_out,
                          track_status from clean.laps""").df()
    pace = race_pace_table(laps)
    con.execute("delete from features.race_pace")
    con.register("_p", pace)
    con.execute("insert into features.race_pace by name select * from _p")
    con.unregister("_p")
    print(f"features.race_pace: {len(pace)} filas, {pace['race_pace_pct'].notna().sum()} con ritmo válido\n")
    print("Por temporada (carreras con ritmo / carreras mojadas / vueltas limpias medias por piloto):")
    print(con.execute("""
        select substr(p.race_id, 1, 4) as temporada, count(distinct p.race_id) as carreras,
               count(distinct case when p.race_pace_pct is not null then p.race_id end) as con_ritmo,
               count(distinct case when p.race_wet then p.race_id end) as mojadas,
               round(avg(p.n_clean_laps), 1) as vueltas_limpias_medias
        from features.race_pace p group by 1 order by 1""").df().to_string(index=False))
    print("\nÚltima carrera con ritmo, de más rápido a más lento (comprobación visual):")
    print(con.execute("""
        select p.race_id, x.team_id, p.driver_id, round(p.race_pace_pct, 3) as ritmo_pct, p.n_clean_laps
        from features.race_pace p join clean.race_results x on x.race_id = p.race_id and x.driver_id = p.driver_id
        where p.race_id = (select max(race_id) from features.race_pace where race_pace_pct is not null)
          and p.race_pace_pct is not null order by p.race_pace_pct""").df().to_string(index=False))
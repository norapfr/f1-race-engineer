"""Crea una plantilla CSV de clasificación con los pilotos de la última carrera disputada, para rellenarla a mano
si la API no responde. Uso: python -m pipelines.quali_template [--out quali.csv] [--db data/f1.duckdb]"""
from __future__ import annotations
import argparse
from storage.db import connect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="quali.csv")
    ap.add_argument("--db", default="data/f1.duckdb")
    a = ap.parse_args()
    con = connect(a.db)
    last = con.execute("""select r.race_id from clean.races r where exists (select 1 from clean.race_results x
                          where x.race_id = r.race_id) order by r.season desc, r."round" desc limit 1""").fetchone()[0]
    d = con.execute("""select x.driver_id, d.name as piloto, t.name as equipo, null::integer as position, null::varchar as q1
                       from clean.race_results x join clean.drivers d on d.driver_id = x.driver_id
                       join clean.teams t on t.team_id = x.team_id where x.race_id = ? order by t.name, d.name""", [last]).df()
    d.to_csv(a.out, index=False)
    print(f"Plantilla creada en {a.out} con los {len(d)} pilotos de {last}.\n"
          "Rellena 'position' (1 = pole) con la clasificación oficial; 'q1' es opcional (p. ej. 1:30.100).\n"
          "Si algún piloto no corre, borra su fila. Luego:\n"
          "  python -m pipelines.predict_upcoming 17 --quali-file quali.csv --circuit marina_bay --why")
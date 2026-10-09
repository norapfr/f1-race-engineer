"""RAW -> CLEAN. Funciones puras (sin red) para poder testearlas con fixtures."""
from __future__ import annotations
import pandas as pd


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def normalize_results(payload: dict) -> dict[str, pd.DataFrame]:
    races, circuits, drivers, teams, results = [], {}, {}, {}, []
    for r in payload["MRData"]["RaceTable"]["Races"]:
        season, rnd = int(r["season"]), int(r["round"])
        race_id = f"{season}_{rnd:02d}"
        c = r["Circuit"]
        circuits[c["circuitId"]] = {
            "circuit_id": c["circuitId"], "name": c["circuitName"],
            "country": c["Location"].get("country"), "locality": c["Location"].get("locality"),
            "lat": float(c["Location"]["lat"]), "lon": float(c["Location"]["long"]),
        }
        races.append({"race_id": race_id, "season": season, "round": rnd, "name": r["raceName"],
                      "circuit_id": c["circuitId"], "race_date": r["date"]})
        for row in r.get("Results", []):
            d, k = row["Driver"], row["Constructor"]
            drivers[d["driverId"]] = {
                "driver_id": d["driverId"], "code": d.get("code"),
                "name": f'{d["givenName"]} {d["familyName"]}', "nationality": d.get("nationality"),
                "number": _int(d.get("permanentNumber")),
            }
            teams[k["constructorId"]] = {"team_id": k["constructorId"], "name": k["name"]}
            results.append({
                "race_id": race_id, "driver_id": d["driverId"], "team_id": k["constructorId"],
                "grid": _int(row.get("grid")), "finish_position": _int(row.get("position")),
                "position_text": row.get("positionText"), "classified": str(row.get("positionText", "")).isdigit(),
                "points": float(row.get("points", 0)), "status": row.get("status"),
                "laps_completed": _int(row.get("laps")),
            })
    out = {
        "races": pd.DataFrame(races), "circuits": pd.DataFrame(circuits.values()),
        "drivers": pd.DataFrame(drivers.values()), "teams": pd.DataFrame(teams.values()),
        "race_results": pd.DataFrame(results),
    }
    if not out["races"].empty:
        out["races"]["race_date"] = pd.to_datetime(out["races"]["race_date"]).dt.date
    return out


def parse_lap_time(s):
    """'1:29.123' -> 89.123; vacío o inválido -> None."""
    if not s:
        return None
    try:
        if ":" in s:
            m, sec = s.split(":")
            return int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return None


def normalize_qualifying(payload: dict) -> pd.DataFrame:
    rows = []
    for r in payload["MRData"]["RaceTable"]["Races"]:
        race_id = f"{int(r['season'])}_{int(r['round']):02d}"
        for q in r.get("QualifyingResults", []):
            rows.append({
                "race_id": race_id, "driver_id": q["Driver"]["driverId"],
                "team_id": q["Constructor"]["constructorId"], "position": _int(q.get("position")),
                "q1_s": parse_lap_time(q.get("Q1")), "q2_s": parse_lap_time(q.get("Q2")),
                "q3_s": parse_lap_time(q.get("Q3")),
            })
    return pd.DataFrame(rows)
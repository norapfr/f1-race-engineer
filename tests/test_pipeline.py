from ingestion.jolpica import merge_race_pages
from ingestion.normalize import normalize_results
from pipelines.ingest_results import load_payload
from storage.db import connect


def _race(results):
    return {"season": "2024", "round": "1", "raceName": "Bahrain Grand Prix", "date": "2024-03-02",
            "Circuit": {"circuitId": "bahrain", "circuitName": "Bahrain International Circuit",
                        "Location": {"lat": "26.0325", "long": "50.5106", "locality": "Sakhir", "country": "Bahrain"}},
            "Results": results}


def _row(pos, pos_text, drv, team, grid, pts, status="Finished", laps="57"):
    return {"position": pos, "positionText": pos_text, "points": pts, "grid": grid, "laps": laps, "status": status,
            "Driver": {"driverId": drv, "code": drv[:3].upper(), "givenName": "A", "familyName": drv,
                       "nationality": "X", "permanentNumber": "1"},
            "Constructor": {"constructorId": team, "name": team.title()}}


def _page(results, total=3):
    return {"MRData": {"total": str(total), "RaceTable": {"Races": [_race(results)]}}}


PAGES = [_page([_row("1", "1", "max", "red_bull", "1", "25"), _row("2", "2", "perez", "red_bull", "5", "18")]),
         _page([_row("20", "R", "zhou", "sauber", "0", "0", status="Engine", laps="10")])]


def test_merge_pages_joins_split_race():
    merged = merge_race_pages(PAGES)
    races = merged["MRData"]["RaceTable"]["Races"]
    assert len(races) == 1 and len(races[0]["Results"]) == 3


def test_normalize_and_dnf_flags():
    t = normalize_results(merge_race_pages(PAGES))
    rr = t["race_results"].set_index("driver_id")
    assert rr.loc["max", "classified"] and not rr.loc["zhou", "classified"]
    assert rr.loc["zhou", "grid"] == 0 and rr.loc["zhou", "status"] == "Engine"
    assert t["races"].loc[0, "race_id"] == "2024_01"


def test_load_is_idempotent():
    con = connect()
    payload = merge_race_pages(PAGES)
    load_payload(con, payload)
    load_payload(con, payload)  # segunda carga: no debe duplicar
    assert con.execute("select count(*) from clean.race_results").fetchone()[0] == 3
    assert con.execute("select count(*) from clean.races").fetchone()[0] == 1

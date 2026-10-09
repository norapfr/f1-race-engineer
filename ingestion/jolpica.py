"""Cliente Jolpica-F1 (sucesor de Ergast): resultados, clasificación, circuitos, standings, pit stops."""
from __future__ import annotations
import time
import requests

BASE = "https://api.jolpi.ca/ergast/f1"


class JolpicaClient:
    def __init__(self, min_interval_s: float = 1.0, max_retries: int = 5, session=None):
        self.min_interval_s = min_interval_s  # cortesía con el servicio; ver límites vigentes en su documentación
        self.max_retries = max_retries
        self.http = session or requests.Session()
        self._last = 0.0

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.http.get(url, params=params, timeout=30)
            self._last = time.monotonic()
            if r.status_code == 429:
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Jolpica: demasiados 429 en {url}")

    def paged(self, path: str, limit: int = 100) -> list[dict]:
        pages, offset = [], 0
        while True:
            js = self._get(f"{BASE}/{path}.json", {"limit": limit, "offset": offset})
            pages.append(js)
            total = int(js["MRData"]["total"])
            offset += limit
            if offset >= total:
                return pages

    def season_results(self, season: int) -> dict:
        return merge_race_pages(self.paged(f"{season}/results"))

    def season_qualifying(self, season: int) -> dict:
        return merge_race_pages(self.paged(f"{season}/qualifying"), list_key="QualifyingResults")


def merge_race_pages(pages: list[dict], list_key: str = "Results") -> dict:
    """La paginación es por filas, así que una misma carrera puede aparecer partida en dos páginas."""
    merged: dict[tuple, dict] = {}
    for js in pages:
        for race in js["MRData"]["RaceTable"]["Races"]:
            k = (race["season"], race["round"])
            if k not in merged:
                merged[k] = {**race, list_key: list(race.get(list_key, []))}
            else:
                merged[k][list_key].extend(race.get(list_key, []))
    races = sorted(merged.values(), key=lambda r: (int(r["season"]), int(r["round"])))
    return {"MRData": {"RaceTable": {"Races": races}}}

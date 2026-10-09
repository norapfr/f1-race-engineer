"""Baselines de la Fase 2: P(win), P(podio) y posición esperada a partir de UNA sola señal pre-carrera
(parrilla, clasificación o standings). Se ajustan solo con temporadas anteriores a la evaluada."""
from __future__ import annotations
import numpy as np
import pandas as pd

MAX_SLOT = 20
STRENGTH = 5.0  # fuerza del prior (en "carreras equivalentes") para suavizar slots con pocos datos


def load(con) -> pd.DataFrame:
    df = con.execute("""
        select r.race_id, r.season, r."round", x.driver_id, x.team_id, x.grid, x.finish_position, x.points,
               q.position as quali_pos
        from clean.race_results x
        join clean.races r on r.race_id = x.race_id
        left join clean.qualifying_results q on q.race_id = x.race_id and q.driver_id = x.driver_id
        where x.finish_position is not null
        order by r.season, r."round" """).df()
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["season", "round"], kind="stable").reset_index(drop=True)
    d["field"] = d.groupby("race_id")["driver_id"].transform("count")
    d["grid_eff"] = np.where(d["grid"] > 0, d["grid"], d["field"])         # pit lane (0) = último
    d["quali_slot"] = d["quali_pos"].fillna(d["grid_eff"])                  # sin tiempo -> usa la parrilla
    # Standings ANTES de la carrera: puntos de la temporada hasta la carrera anterior;
    # en la primera ronda, total de la temporada previa. (Solo puntos de carrera, sin sprint.)
    d["pts_before"] = d.groupby(["season", "driver_id"])["points"].cumsum() - d["points"]
    prev = d.groupby(["season", "driver_id"])["points"].sum().rename("prev_total").reset_index()
    prev["season"] += 1
    d = d.merge(prev, on=["season", "driver_id"], how="left")
    d["prev_total"] = d["prev_total"].fillna(0.0)
    base = np.where(d["round"] == 1, d["prev_total"], d["pts_before"])
    d["standings_slot"] = pd.Series(base, index=d.index).groupby(d["race_id"]).rank(ascending=False, method="min")
    return d


class Uniform:
    """Referencia de azar: todos los pilotos de la carrera igual de probables."""
    def fit(self, tr):
        return self

    def predict(self, te):
        f = te["field"].to_numpy(float)
        return 1 / f, np.minimum(3, f) / f, (f + 1) / 2


class SlotBaseline:
    """Estima, con temporadas pasadas, P(win), P(podio) y posición media según el 'slot' (parrilla, quali o standings)."""
    def __init__(self, slot_col: str):
        self.slot_col = slot_col

    def _slot(self, df) -> np.ndarray:
        return df[self.slot_col].fillna((MAX_SLOT + 1) / 2).clip(1, MAX_SLOT).round().astype(int).to_numpy()
    
    def fit(self, tr):
        g = pd.DataFrame({"slot": self._slot(tr), "win": (tr["finish_position"] == 1).to_numpy(float),
                          "pod": (tr["finish_position"] <= 3).to_numpy(float),
                          "fin": tr["finish_position"].to_numpy(float)})
        glob = g[["win", "pod", "fin"]].mean()
        agg = g.groupby("slot").agg(n=("win", "size"), win=("win", "sum"), pod=("pod", "sum"), fin=("fin", "sum"))
        agg = agg.reindex(range(1, MAX_SLOT + 1)).fillna(0.0)
        self.table = pd.DataFrame({c: (agg[k] + STRENGTH * glob[k]) / (agg["n"] + STRENGTH)
                                   for c, k in [("p_win", "win"), ("p_pod", "pod"), ("e_fin", "fin")]})
        return self

    def predict(self, te):
        s = self._slot(te)
        pw = self.table["p_win"].reindex(s).to_numpy()
        pw = pw / pd.Series(pw).groupby(te["race_id"].to_numpy()).transform("sum").to_numpy()  # suma 1 por carrera
        return pw, self.table["p_pod"].reindex(s).to_numpy(), self.table["e_fin"].reindex(s).to_numpy()


def winner_acc(te: pd.DataFrame, p: np.ndarray) -> float:
    """Acierto del ganador; los empates se reparten (esperanza), así el azar no sale favorecido por el orden."""
    d = te[["race_id", "finish_position"]].assign(p=np.round(p, 12), y=(te["finish_position"] == 1).astype(float))
    top = d[d["p"] == d.groupby("race_id")["p"].transform("max")]
    return float(top.groupby("race_id")["y"].mean().mean())


def metrics(te: pd.DataFrame, pw, pp, ef) -> dict:
    yw = (te["finish_position"] == 1).to_numpy(float)
    yp = (te["finish_position"] <= 3).to_numpy(float)
    y = te["finish_position"].to_numpy(float)
    pwc = np.clip(pw, 1e-6, 1 - 1e-6)
    return {"n_races": te["race_id"].nunique(), "winner_acc": winner_acc(te, pw),
            "brier_win": float(np.mean((pw - yw) ** 2)),
            "logloss_win": float(-np.mean(yw * np.log(pwc) + (1 - yw) * np.log(1 - pwc))),
            "brier_podium": float(np.mean((pp - yp) ** 2)),
            "mae_finish": float(np.mean(np.abs(ef - y))), "rmse_finish": float(np.sqrt(np.mean((ef - y) ** 2)))}
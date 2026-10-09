"""Tabla de features pre-carrera por (race_id, driver_id).
Regla: todo agregado histórico usa SOLO carreras anteriores (shift antes de la ventana)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .availability import check_features
from .baselines import prepare
from .temporal import prior_ewm

HL_FORM, HL_DNF = 3.0, 5.0   # semividas (en carreras) de la media exponencial
ID_COLS = ["race_id", "season", "round", "driver_id", "team_id", "circuit_id"]
TARGETS = ["target_win", "target_podium", "target_top5", "target_top10", "target_finish", "target_points"]
BASE_FEATURES = [
    # parrilla / clasificación (disponibles al terminar la clasificación)
    "grid_eff", "grid_field_size", "quali_slot", "quali_q1_gap_pct", "quali_teammate_delta",
    # piloto: forma reciente y perfil
    "prior_driver_finish_ewm", "prior_driver_points_ewm", "prior_driver_dnf_ewm", "prior_driver_gain_ewm",
    "prior_driver_n_races", "prior_driver_form_rank", "prior_standings_rank",
    # piloto x circuito
    "prior_driver_circuit_finish_mean", "prior_driver_circuit_n",
    # equipo: forma y fiabilidad
    "prior_team_points_ewm", "prior_team_finish_ewm", "prior_team_dnf_ewm", "prior_team_form_rank",
]
# ritmo de carrera de carreras anteriores (FastF1); menor % = más rápido que el campo
PACE_FEATURES = ["prior_driver_pace_ewm", "prior_driver_pace_vs_team", "prior_team_pace_ewm",
                 "prior_team_pace_gap", "prior_team_pace_rank"]
FEATURES = BASE_FEATURES + PACE_FEATURES


def load_raw(con) -> pd.DataFrame:
    return con.execute("""
        select r.race_id, r.season, r."round", r.circuit_id, x.driver_id, x.team_id, x.grid, x.finish_position,
               x.points, x.classified, q.position as quali_pos, q.q1_s, p.race_pace_pct
        from clean.race_results x
        join clean.races r on r.race_id = x.race_id
        left join clean.qualifying_results q on q.race_id = x.race_id and q.driver_id = x.driver_id
        left join features.race_pace p on p.race_id = x.race_id and p.driver_id = x.driver_id
        where x.finish_position is not null
        order by r.season, r."round" """).df()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    if "race_pace_pct" not in raw.columns:   # sin datos de ritmo: las features de ritmo quedan en NaN
        raw = raw.assign(race_pace_pct=np.nan)
    d = prepare(raw)  # añade field, grid_eff, quali_slot, standings_slot (ya sin información de la propia carrera)
    d["dnf"] = (~d["classified"].astype(bool)).astype(float)
    d["gain"] = d["grid_eff"] - d["finish_position"]

    # --- clasificación (conocida antes de la carrera)
    d["grid_field_size"] = d["field"]
    d["quali_q1_gap_pct"] = (d["q1_s"] / d.groupby("race_id")["q1_s"].transform("min") - 1) * 100
    g = d.groupby(["race_id", "team_id"])["quali_slot"]
    mates = (g.transform("count") - 1).replace(0, np.nan)
    d["quali_teammate_delta"] = d["quali_slot"] - (g.transform("sum") - d["quali_slot"]) / mates
    d["prior_standings_rank"] = d["standings_slot"]

    # --- piloto (solo carreras previas)
    d["prior_driver_finish_ewm"] = prior_ewm(d, "driver_id", "finish_position", HL_FORM)
    d["prior_driver_points_ewm"] = prior_ewm(d, "driver_id", "points", HL_FORM)
    d["prior_driver_dnf_ewm"] = prior_ewm(d, "driver_id", "dnf", HL_DNF)
    d["prior_driver_gain_ewm"] = prior_ewm(d, "driver_id", "gain", HL_FORM)
    d["prior_driver_n_races"] = d.groupby("driver_id").cumcount()
    gc = d.groupby(["driver_id", "circuit_id"])
    d["prior_driver_pace_ewm"] = prior_ewm(d, "driver_id", "race_pace_pct", HL_FORM)
    d["prior_driver_circuit_finish_mean"] = gc["finish_position"].transform(lambda s: s.shift(1).expanding().mean())
    d["prior_driver_circuit_n"] = gc.cumcount()

    # --- equipo (se agrega por carrera y se desplaza igual)
    tr = (d.groupby(["race_id", "team_id", "season", "round"], as_index=False)
            .agg(team_points=("points", "sum"), team_finish=("finish_position", "mean"), team_dnf=("dnf", "mean"),
                  team_pace=("race_pace_pct", "mean"))
            .sort_values(["season", "round"], kind="stable").reset_index(drop=True))
    tr["prior_team_points_ewm"] = prior_ewm(tr, "team_id", "team_points", HL_FORM)
    tr["prior_team_finish_ewm"] = prior_ewm(tr, "team_id", "team_finish", HL_FORM)
    tr["prior_team_dnf_ewm"] = prior_ewm(tr, "team_id", "team_dnf", HL_DNF)
    tr["prior_team_pace_ewm"] = prior_ewm(tr, "team_id", "team_pace", HL_FORM)
    d = d.merge(tr[["race_id", "team_id", "prior_team_points_ewm", "prior_team_finish_ewm", "prior_team_dnf_ewm",
                    "prior_team_pace_ewm"]], on=["race_id", "team_id"], how="left")
    d["prior_driver_pace_vs_team"] = d["prior_driver_pace_ewm"] - d["prior_team_pace_ewm"]
    d["prior_team_pace_gap"] = d["prior_team_pace_ewm"] - d.groupby("race_id")["prior_team_pace_ewm"].transform("min")
    d["prior_team_pace_rank"] = d.groupby("race_id")["prior_team_pace_ewm"].rank(method="dense")  # 1 = equipo más rápido

    # --- rankings dentro de la carrera (comparan valores PREVIOS de todos los pilotos de la misma carrera)
    d["prior_driver_form_rank"] = d.groupby("race_id")["prior_driver_points_ewm"].rank(ascending=False, method="min")
    d["prior_team_form_rank"] = d.groupby("race_id")["prior_team_points_ewm"].rank(ascending=False, method="min")

    # --- objetivos (POST_RACE: solo para entrenar/evaluar, nunca como feature)
    fp = d["finish_position"]
    d["target_win"], d["target_podium"] = (fp == 1).astype(int), (fp <= 3).astype(int)
    d["target_top5"], d["target_top10"] = (fp <= 5).astype(int), (fp <= 10).astype(int)
    d["target_finish"], d["target_points"] = fp, d["points"]

    check_features(FEATURES)  # falla si alguna feature no es pre-carrera o no está registrada
    d = d.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)   # orden independiente de la entrada
    return d[ID_COLS + FEATURES + TARGETS]
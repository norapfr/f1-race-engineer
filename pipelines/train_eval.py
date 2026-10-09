"""Entrena y evalúa los modelos con validación temporal expansiva, los compara con los baselines y mide
(ablación) cuánto aportan las features de ritmo de carrera.
Uso: python -m pipelines.train_eval data/f1.duckdb"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from features.baselines import SlotBaseline, metrics
from features.build import BASE_FEATURES, FEATURES, build_features, load_raw
from features.temporal import assert_temporal_split
from models.evaluation import bootstrap_diff, calibration_table, ece, race_terms
from models.race_model import FAMILIES, predict_season
from storage.db import connect

MIN_TRAIN_SEASONS = 3
REFERENCE = "3_quali"
BASELINES = {"1_grid": "grid_eff", REFERENCE: "quali_slot"}
SETS = {"": BASE_FEATURES, "_pace": FEATURES}   # sufijo -> conjunto de features
EXCLUDE_SEASON = 2023   # análisis de sensibilidad: temporada dominada por un solo coche


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["field"], df["finish_position"] = df["grid_field_size"], df["target_finish"]  # etiqueta: solo para evaluar
    return df


def _select(pooled: dict, exclude=None) -> dict:
    return {n: (p if exclude is None else p[p["season"] != exclude]) for n, p in pooled.items()}


def _row(name: str, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    d_ll, d_ae = bootstrap_diff(a, b, "ll"), bootstrap_diff(a, b, "ae")
    return {"modelo": name, "n_races": len(a), "d_logloss_win": d_ll[0], "ic95_lo": d_ll[1], "ic95_hi": d_ll[2],
            "d_mae_pos": d_ae[0], "ic95_lo_mae": d_ae[1], "ic95_hi_mae": d_ae[2]}


def _terms(p: pd.DataFrame) -> pd.DataFrame:
    return race_terms(p, p["pw"], p["ef"])


def _diffs(pooled: dict, exclude=None) -> pd.DataFrame:
    sel = _select(pooled, exclude)
    ref = _terms(sel[REFERENCE])
    return pd.DataFrame([_row(n, _terms(p), ref) for n, p in sel.items() if n != REFERENCE])


def _ablation(pooled: dict, exclude=None) -> pd.DataFrame:
    """Cada modelo con ritmo frente al mismo modelo sin ritmo (negativo = el ritmo ayuda)."""
    sel = _select(pooled, exclude)
    return pd.DataFrame([_row(f"{f.name}_pace vs {f.name}", _terms(sel[f.name + "_pace"]), _terms(sel[f.name]))
                         for f in FAMILIES])


def run(feat: pd.DataFrame, exclude_season: int = EXCLUDE_SEASON) -> dict:
    df = _prepare(feat)
    seasons = sorted(df["season"].unique())
    parts, cfgs = {}, []
    for t in seasons[MIN_TRAIN_SEASONS:]:
        tr = df[df["season"] < t]
        te0 = df[df["season"] == t].reset_index(drop=True)
        assert_temporal_split((tr["season"] * 100 + tr["round"]).tolist(), (te0["season"] * 100 + te0["round"]).tolist())
        for name, col in BASELINES.items():
            pw, pp, ef = SlotBaseline(col).fit(tr).predict(te0)
            parts.setdefault(name, []).append(te0.assign(pw=pw, pp=pp, ef=ef))
        for fam in FAMILIES:
            for suffix, feats in SETS.items():
                te, P, cfg = predict_season(fam, df, t, feats)
                parts.setdefault(fam.name + suffix, []).append(te.assign(pw=P["p_win"], pp=P["p_pod"], ef=P["ef"]))
                cfgs.append({"season": t, "modelo": fam.name + suffix, "config": str(cfg)})
        # conjunto exploratorio: promedio de los dos modelos con todas las features
    parts["ens"] = [a.assign(pw=(a["pw"] + b["pw"]) / 2, pp=(a["pp"] + b["pp"]) / 2, ef=(a["ef"] + b["ef"]) / 2)
                    for a, b in zip(parts["logit_pace"], parts["lgbm_pace"])]
    rows = []
    for name, ps in parts.items():
        for t, p in [(p["season"].iloc[0], p) for p in ps] + [("todas", pd.concat(ps, ignore_index=True))]:
            m = metrics(p, p["pw"], p["pp"], p["ef"])
            rows.append({"season": t, "modelo": name, **m, "ece_win": ece(p["target_win"], p["pw"]),
                         "ece_podium": ece(p["target_podium"], p["pp"])})
    pooled = {n: pd.concat(ps, ignore_index=True) for n, ps in parts.items()}
    res = pd.DataFrame(rows)
    ml = [f.name + s for f in FAMILIES for s in SETS] + ["ens"]
    best = res[(res["season"] == "todas") & res["modelo"].isin(ml)].sort_values("logloss_win").iloc[0]["modelo"]
    return {"metrics": res, "diffs": _diffs(pooled), "diffs_sin": _diffs(pooled, exclude_season),
            "ablation": _ablation(pooled), "ablation_sin": _ablation(pooled, exclude_season),
            "exclude_season": exclude_season, "configs": pd.DataFrame(cfgs), "best": best,
            "calib_win": calibration_table(pooled[best]["target_win"], pooled[best]["pw"]),
            "calib_podium": calibration_table(pooled[best]["target_podium"], pooled[best]["pp"])}


if __name__ == "__main__":
    out = run(build_features(load_raw(connect(sys.argv[1] if len(sys.argv) > 1 else "data/f1.duckdb"))))
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out["metrics"].to_csv("data/processed/model_eval.csv", index=False)
    m = out["metrics"]
    cols = ["modelo", "n_races", "winner_acc", "logloss_win", "brier_win", "brier_podium", "mae_finish",
            "ece_win", "ece_podium"]
    print("== Métricas agrupadas (todas las temporadas de test) ==")
    print(m[m["season"] == "todas"][cols].round(4).to_string(index=False))
    print("\n== Log loss de victoria por temporada ==")
    print(m[m["season"] != "todas"].pivot(index="modelo", columns="season", values="logloss_win").round(4).to_string())
    print(f"\n== Diferencia frente a {REFERENCE} (negativo = mejor; IC95% por bootstrap de carreras) ==")
    print(out["diffs"].round(4).to_string(index=False))
    print(f"\n== Lo mismo SIN la temporada {out['exclude_season']} ==")
    print(out["diffs_sin"].round(4).to_string(index=False))
    print("\n== ABLACIÓN: ¿ayuda el ritmo de carrera? (negativo = sí) ==")
    print(out["ablation"].round(4).to_string(index=False))
    print(f"\n== Ablación SIN la temporada {out['exclude_season']} ==")
    print(out["ablation_sin"].round(4).to_string(index=False))
    for title, key in [("victoria", "calib_win"), ("podio", "calib_podium")]:
        print(f"\n== Calibración de {title}: {out['best']} ==")
        print(out[key].round(4).to_string(index=False))
    print("\n== Configuración elegida por temporada ==")
    print(out["configs"].to_string(index=False))
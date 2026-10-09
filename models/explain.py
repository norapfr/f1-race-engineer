"""Explicabilidad: qué factores suben o bajan la probabilidad de ganar de cada piloto en una carrera.
Contribuciones en escala log-odds, promediadas entre la regresión logística (coeficiente x valor estandarizado)
y LightGBM (valores SHAP, pred_contrib). Es una aproximación: el modelo final normaliza las probabilidades por
carrera y promedia en probabilidad, no en log-odds. Cada contribución es el efecto frente a un piloto medio."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.build import FEATURES
from models.predict import _select_config
from models.race_model import FAMILIES

LABELS = {
    "grid_eff": "posición de salida", "grid_field_size": "tamaño de la parrilla",
    "quali_slot": "posición en clasificación", "quali_q1_gap_pct": "distancia a la mejor vuelta de Q1",
    "quali_teammate_delta": "clasificación frente a su compañero",
    "prior_driver_finish_ewm": "resultados recientes del piloto", "prior_driver_points_ewm": "puntos recientes del piloto",
    "prior_driver_dnf_ewm": "abandonos recientes del piloto", "prior_driver_gain_ewm": "posiciones que suele ganar en carrera",
    "prior_driver_n_races": "experiencia del piloto", "prior_driver_form_rank": "ranking de forma del piloto",
    "prior_standings_rank": "posición en el campeonato",
    "prior_driver_circuit_finish_mean": "historial del piloto en este circuito",
    "prior_driver_circuit_n": "carreras previas en este circuito",
    "prior_team_points_ewm": "puntos recientes del equipo", "prior_team_finish_ewm": "resultados recientes del equipo",
    "prior_team_dnf_ewm": "fiabilidad reciente del equipo", "prior_team_form_rank": "ranking de forma del equipo",
    "prior_driver_pace_ewm": "ritmo de carrera reciente del piloto",
    "prior_driver_pace_vs_team": "ritmo del piloto frente a su equipo",
    "prior_team_pace_ewm": "ritmo de carrera reciente del equipo",
    "prior_team_pace_gap": "distancia del equipo al más rápido", "prior_team_pace_rank": "ranking de ritmo del equipo",
}

# Los factores de un mismo grupo están muy correlacionados: el modelo reparte entre ellos el efecto de forma arbitraria
# (incluso con signos opuestos). Sumados por grupo, la lectura es estable.
GROUPS = {
    "clasificación y parrilla": ["grid_eff", "grid_field_size", "quali_slot", "quali_q1_gap_pct", "quali_teammate_delta"],
    "forma del piloto": ["prior_driver_finish_ewm", "prior_driver_points_ewm", "prior_driver_dnf_ewm",
                         "prior_driver_gain_ewm", "prior_driver_n_races", "prior_driver_form_rank", "prior_standings_rank"],
    "ritmo de carrera reciente": ["prior_driver_pace_ewm", "prior_driver_pace_vs_team", "prior_team_pace_ewm",
                                  "prior_team_pace_gap", "prior_team_pace_rank"],
    "forma y fiabilidad del equipo": ["prior_team_points_ewm", "prior_team_finish_ewm", "prior_team_dnf_ewm",
                                      "prior_team_form_rank"],
    "historial en el circuito": ["prior_driver_circuit_finish_mean", "prior_driver_circuit_n"],
}


def logit_contrib(pipe, X: pd.DataFrame):
    """Contribución de cada feature = coeficiente x valor estandarizado (suma el indicador de dato ausente)."""
    imp, sc, lr = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["m"]
    Z = sc.transform(imp.transform(X))
    names = [n.replace("missingindicator_", "") for n in imp.get_feature_names_out(list(X.columns))]
    C = pd.DataFrame(Z * lr.coef_[0], columns=names, index=X.index).T.groupby(level=0).sum().T
    return C.reindex(columns=list(X.columns)).fillna(0.0), float(lr.intercept_[0])


def gbm_contrib(model, X: pd.DataFrame):
    """Valores SHAP de LightGBM, promediados entre las semillas. Devuelve (contribuciones, valor base)."""
    A = np.mean([m.predict(X, pred_contrib=True) for m in model.models], axis=0)
    return pd.DataFrame(A[:, :-1], columns=X.columns, index=X.index), A[:, -1]


def explain_race(df: pd.DataFrame, race_id: str, features=None) -> pd.DataFrame:
    """Contribuciones medias (log-odds de ganar) por piloto y feature. Entrena solo con carreras anteriores."""
    features = list(features or FEATURES)
    te = df[df["race_id"] == race_id].reset_index(drop=True)
    if te.empty:
        raise ValueError(f"Carrera desconocida: {race_id}")
    season, rnd = int(te["season"].iloc[0]), int(te["round"].iloc[0])
    past = df[(df["season"] * 100 + df["round"]) < season * 100 + rnd]
    if not {season - 1, season - 2} <= set(past["season"].unique()):
        raise ValueError("Hacen falta al menos dos temporadas completas anteriores a la carrera.")
    parts = []
    for fam in FAMILIES:
        cfg = _select_config(fam, past, season, features)
        clf = fam.make_clf(cfg).fit(past[features], past["target_win"])
        C, _ = logit_contrib(clf, te[features]) if fam.name == "logit" else gbm_contrib(clf, te[features])
        parts.append(C.reindex(columns=features).fillna(0.0).reset_index(drop=True))
    C = sum(parts) / len(parts)
    C.index = te["driver_id"].to_numpy()
    return C


def explain_table(C: pd.DataFrame, names, order, top: int = 3, min_effect: float = 0.1) -> pd.DataFrame:
    """Para cada piloto de `order`: factores que más suben y que más bajan su probabilidad (efecto multiplicativo
    sobre las probabilidades de ganar, frente a un piloto medio)."""
    fmt = lambda s: "; ".join(f"{LABELS.get(k, k)} x{np.exp(v):.1f}" for k, v in s.items()) or "-"
    rows = []
    for d in order:
        s = C.loc[d]
        rows.append({"piloto": names.get(d, d), "sube": fmt(s[s > min_effect].nlargest(top)),
                     "baja": fmt(s[s < -min_effect].nsmallest(top))})
    return pd.DataFrame(rows)


def group_contrib(C: pd.DataFrame) -> pd.DataFrame:
    """Suma de contribuciones (log-odds) por grupo de factores."""
    return pd.DataFrame({g: C.reindex(columns=cols).fillna(0.0).sum(axis=1) for g, cols in GROUPS.items()})


def explain_groups(C: pd.DataFrame, names, order) -> pd.DataFrame:
    """Matriz piloto x grupo con el efecto multiplicativo (x) sobre las probabilidades de ganar frente a un piloto medio."""
    G = np.exp(group_contrib(C).loc[list(order)])
    G.insert(0, "piloto", [names.get(d, d) for d in order])
    return G.reset_index(drop=True)
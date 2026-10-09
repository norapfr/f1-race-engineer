"""Agregados temporales SIEMPRE con información estrictamente anterior (shift(1) antes de la ventana)."""
from __future__ import annotations
import pandas as pd
from .availability import LeakageError

ORDER = ["season", "round"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(ORDER, kind="stable")


def prior_window_mean(df: pd.DataFrame, group: str, value: str, window: int) -> pd.Series:
    """Media de las `window` carreras previas del grupo (excluye la carrera actual)."""
    d = _sorted(df)
    return d.groupby(group)[value].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def prior_ewm(df: pd.DataFrame, group: str, value: str, halflife: float) -> pd.Series:
    """Media móvil exponencial de carreras previas (excluye la actual)."""
    d = _sorted(df)
    return d.groupby(group)[value].transform(lambda s: s.shift(1).ewm(halflife=halflife, min_periods=1).mean())


def race_key(season: int, rnd: int) -> int:
    return season * 100 + rnd


def assert_temporal_split(train_keys, test_keys) -> None:
    if max(train_keys) >= min(test_keys):
        raise LeakageError("El train contiene carreras iguales o posteriores al test.")


def walk_forward(seasons: list[int], min_train: int = 3):
    """Expanding window: train=[s0..sk], val=sk+1, test=sk+2. Ej.: 2018-21 / 2022 / 2023, luego 2018-22 / 2023 / 2024..."""
    seasons = sorted(seasons)
    for i in range(min_train, len(seasons) - 1):
        yield {"train": seasons[:i], "val": seasons[i], "test": seasons[i + 1]}

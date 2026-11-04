"""Forecaster protocol shared by baselines, ML models and the backtest harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

PRED_COLS = ["p10", "p50", "p90"]


@runtime_checkable
class Forecaster(Protocol):
    """One instance = one horizon.

    fit:     rows from features.make_features (y is never NaN)
    predict: returns a DataFrame with the same index as `test` and columns p10, p50, p90.
             p50 is the point forecast; models without intervals put NaN in p10/p90.
    """

    name: str

    def fit(self, train: pd.DataFrame) -> Forecaster: ...

    def predict(self, test: pd.DataFrame) -> pd.DataFrame: ...

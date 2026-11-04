"""LightGBM quantile forecaster with a split-conformal (CQR) interval adjustment.

Three gradient-boosted models predict the 10th, 50th and 90th percentile of next-days PM2.5.

Why log1p(y): burning-season days can be ten times the rainy-season level. Trees fitted on the raw
scale spend most splits on those spikes; on the log scale errors are closer to relative errors.
Quantiles are invariant under monotone transforms, so ``expm1(q_alpha(log1p y)) = q_alpha(y)``
and nothing is lost by transforming back.

Why conformal calibration: quantile boosting is usually over-confident (the p10-p90 band covers
less than 80% out of sample) because the trees fit the training rows too well. Conformalized
quantile regression (Romano et al., 2019) fixes that with a held-out set: fit on the early rows,
measure on the latest ~15% how far the truth falls below p10 and above p90, and move each edge by
the amount that makes it hold ~10% of the misses. The models are then refitted on all rows and
the learned offsets applied to their band. Each tail is calibrated separately so p10 and p90 keep
their meaning on skewed data, where misses are mostly on the high side. The hold-out is the most
recent stretch, not a random sample, so the offsets reflect true forward-in-time errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from thaiair import config
from thaiair.models.base import PRED_COLS
from thaiair.models.registry import as_known_categories, categorical_columns, resolve_feature_cols, tail_by_target_date

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "colsample_bytree": 0.8,  # = feature_fraction
    "reg_lambda": 1.0,
    "n_jobs": config.N_THREADS,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,  # deterministic and avoids the col/row-wise auto-test message
    "verbose": -1,
}
MIN_CALIBRATION_ROWS = 100  # below this the offsets are noise; skip calibration instead


class LightGBMQuantile:
    """p10 / p50 / p90 of PM2.5 for one horizon from three LightGBM quantile regressors.

    Args:
        horizon: days ahead (1-3); only used to pick the default feature columns.
        feature_cols: columns of the ``make_features`` table to use; default
            ``thaiair.features.feature_columns(horizon)``.
        params: overrides for ``DEFAULT_PARAMS`` (passed to ``lightgbm.LGBMRegressor``).
        calibrate: apply the split-conformal band adjustment described in the module docstring.
        cal_frac: share of the latest target dates held out for calibration.
    """

    name = "lgbm"

    def __init__(
        self,
        horizon: int,
        feature_cols: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        calibrate: bool = True,
        cal_frac: float = 0.15,
    ) -> None:
        self.horizon = horizon
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.calibrate = calibrate
        self.cal_frac = cal_frac
        self.quantiles = config.QUANTILES
        self.models_: list[lgb.LGBMRegressor] = []
        self.feature_cols_: list[str] = []
        self.categories_: dict[str, list[Any]] = {}
        self.calibration_: dict[str, float] = {"offset_lo": 0.0, "offset_hi": 0.0, "n_cal": 0}

    # ----------------------------------------------------------------------- public API

    def fit(self, train: pd.DataFrame) -> LightGBMQuantile:
        """Fit on rows of ``make_features`` with non-NaN ``y`` (calibrate first, then refit on all)."""
        self.feature_cols_ = resolve_feature_cols(self.horizon, self.feature_cols)
        self.categories_ = {
            c: sorted(train[c].dropna().unique().tolist()) for c in categorical_columns(self.feature_cols_)
        }
        X = self._matrix(train)
        y = np.log1p(np.clip(train["y"].to_numpy(dtype=float), 0, None))

        self.calibration_ = {"offset_lo": 0.0, "offset_hi": 0.0, "n_cal": 0}
        if self.calibrate:
            cal = tail_by_target_date(train["target_date"], self.cal_frac)
            if MIN_CALIBRATION_ROWS <= cal.sum() < len(cal):
                models = self._fit_models(X[~cal], y[~cal])
                self.calibration_ = _conformal_offsets(_raw_quantiles(models, X[cal]), y[cal])

        self.models_ = self._fit_models(X, y)
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Calibrated p10 / p50 / p90 in µg/m³, same index as ``test``."""
        if not self.models_:
            raise RuntimeError("LightGBMQuantile.predict called before fit")
        q = _raw_quantiles(self.models_, self._matrix(test))
        lo = q[:, 0] - self.calibration_["offset_lo"]
        hi = q[:, 2] + self.calibration_["offset_hi"]
        mid = q[:, 1]
        # A negative offset (band too wide) must not push an edge past the median.
        out = np.column_stack([np.minimum(lo, mid), mid, np.maximum(hi, mid)])
        return pd.DataFrame(np.clip(np.expm1(out), 0, None), index=test.index, columns=PRED_COLS)

    def feature_importance(self) -> pd.Series:
        """Total split gain per feature of the p50 model, largest first."""
        if not self.models_:
            raise RuntimeError("LightGBMQuantile.feature_importance called before fit")
        booster = self.models_[1].booster_
        gain = booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=booster.feature_name(), name="gain").sort_values(ascending=False)

    # ----------------------------------------------------------------------- internals

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature frame with categoricals pinned to the categories seen in fit.

        Fixed categories make the integer codes LightGBM sees identical at fit and predict time;
        an unseen city becomes NaN (LightGBM's "missing" branch) instead of a wrong code.
        """
        X = df[self.feature_cols_].copy()
        for col, cats in self.categories_.items():
            X[col] = as_known_categories(X[col], cats)
        for col in X.columns.difference(list(self.categories_)):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        return X

    def _fit_models(self, X: pd.DataFrame, y: np.ndarray) -> list[lgb.LGBMRegressor]:
        models = []
        for alpha in self.quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **self.params)
            model.fit(X, y, categorical_feature=list(self.categories_) or "auto")
            models.append(model)
        return models


def _raw_quantiles(models: list[lgb.LGBMRegressor], X: pd.DataFrame) -> np.ndarray:
    """(n, 3) log-scale predictions, sorted per row so p10 <= p50 <= p90 (no quantile crossing)."""
    return np.sort(np.column_stack([m.predict(X) for m in models]), axis=1)


def _conformal_offsets(q: np.ndarray, y: np.ndarray, tail: float = 0.1) -> dict[str, float]:
    """Per-tail CQR offsets (log scale) so each edge misses about ``tail`` of held-out rows.

    The score ``lo - y`` is positive when the truth falls below p10. Its ``(1 - tail)`` quantile,
    with the usual ``(n + 1) / n`` finite-sample correction, is how far p10 must move down
    (negative = it may move up). Same for the upper edge with ``y - hi``.
    """
    n = len(y)
    level = min(1.0, (1 - tail) * (n + 1) / n)
    off_lo = float(np.quantile(q[:, 0] - y, level, method="higher"))
    off_hi = float(np.quantile(y - q[:, 2], level, method="higher"))
    raw_cov = float(np.mean((y >= q[:, 0]) & (y <= q[:, 2])))
    return {"offset_lo": off_lo, "offset_hi": off_hi, "n_cal": n, "raw_coverage": raw_cov}

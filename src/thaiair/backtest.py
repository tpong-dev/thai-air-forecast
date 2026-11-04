"""Rolling-origin, expanding-window backtest: the only place where models are scored.

For each fold start M (monthly by default) a fresh forecaster is trained on every row whose
*target* day is before M (so each training label was already observed when the fold begins) and
predicts all rows whose *issue* day falls in the fold. Training on target_date < M rather than
issue_date < M matters: a row issued on M-1 with h=3 has its label on M+2, which is in the future.

Test rows are kept even when `y` is NaN or a model returns NaN (e.g. CAMS before Sep 2022), so the
output shows exactly what was and was not scored; `metrics.paired` then picks like-for-like rows.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from thaiair import config
from thaiair.features import make_features
from thaiair.models.base import PRED_COLS, Forecaster

log = logging.getLogger(__name__)

OUTPUT_COLS = ["model", "city", "horizon", "issue_date", "target_date", "y", *PRED_COLS]
DEFAULT_REFIT = "MS"

Factory = Callable[[int], Forecaster]


def fold_starts(
    test_start: str | pd.Timestamp, test_end: str | pd.Timestamp, freq: str = DEFAULT_REFIT
) -> list[pd.Timestamp]:
    """Refit dates: `test_start` itself, then every `freq` boundary (e.g. month start) up to `test_end`."""
    start, end = pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end).normalize()
    if end < start:
        raise ValueError("test_end is before test_start")
    bounds = pd.date_range(start, end, freq=freq)
    return sorted({start, *bounds})


def _refit_freq(refit: str | Mapping[str, str], model: str) -> str:
    """`refit` is one frequency for all models, or a {model: freq} map (others use monthly)."""
    if isinstance(refit, str):
        return refit
    return refit.get(model, DEFAULT_REFIT)


def _check_predictions(pred: pd.DataFrame, test: pd.DataFrame, model: str) -> pd.DataFrame:
    """Enforce the Forecaster contract so a buggy model fails loudly instead of misaligning rows."""
    missing = [c for c in PRED_COLS if c not in pred.columns]
    if missing:
        raise ValueError(f"{model}.predict() is missing columns {missing}")
    if len(pred) != len(test) or not pred.index.equals(test.index):
        raise ValueError(f"{model}.predict() must return the same index as its input")
    return pred[PRED_COLS].astype(float)


def _run_fold(factory: Factory, horizon: int, train: pd.DataFrame, test: pd.DataFrame, model: str) -> pd.DataFrame:
    """Fit a fresh instance on `train`, predict `test`; an empty training set yields NaN forecasts."""
    if train.empty:
        log.warning(
            "%s h=%d: no training rows before %s, predictions are NaN", model, horizon, test["issue_date"].min()
        )
        return pd.DataFrame(np.nan, index=test.index, columns=PRED_COLS)
    forecaster = factory(horizon).fit(train)
    return _check_predictions(forecaster.predict(test), test, model)


def _run_model(
    model: str, factory: Factory, feats: pd.DataFrame, horizon: int, starts: list[pd.Timestamp], end: pd.Timestamp
) -> pd.DataFrame:
    """All folds of one model at one horizon, concatenated in time order."""
    labelled = feats[feats["y"].notna()]
    out = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end + pd.Timedelta(days=1)
        test = feats[(feats["issue_date"] >= start) & (feats["issue_date"] < stop)]
        if test.empty:
            continue
        train = labelled[labelled["target_date"] < start]
        t0 = time.perf_counter()
        pred = _run_fold(factory, horizon, train, test, model)
        log.info(
            "%-12s h=%d fold %s: train %6d rows, test %4d rows, %.1fs",
            model,
            horizon,
            start.date(),
            len(train),
            len(test),
            time.perf_counter() - t0,
        )
        out.append(pd.concat([test[["city", "horizon", "issue_date", "target_date", "y"]], pred], axis=1))
    if not out:
        return pd.DataFrame(columns=OUTPUT_COLS)
    return pd.concat(out).assign(model=model)[OUTPUT_COLS]


def run(
    panel: pd.DataFrame,
    factories: Mapping[str, Factory],
    horizons: tuple[int, ...] = config.HORIZONS,
    test_start: str | pd.Timestamp = config.TEST_START,
    test_end: str | pd.Timestamp = config.TEST_END,
    refit: str | Mapping[str, str] = DEFAULT_REFIT,
) -> pd.DataFrame:
    """Backtest every model at every horizon; returns the long table `OUTPUT_COLS`.

    `factories` maps a model name to `lambda horizon: Forecaster`. `refit` is a pandas frequency
    for all models or a per-model map such as `{"deep": "QS"}` (quarterly refits save CPU for the
    slow model; unlisted models refit monthly). Deterministic given deterministic forecasters.
    """
    end = pd.Timestamp(test_end).normalize()
    results = []
    for horizon in horizons:
        feats = make_features(panel, horizon)
        for model, factory in factories.items():
            starts = fold_starts(test_start, end, _refit_freq(refit, model))
            log.info("backtest %s h=%d: %d folds from %s", model, horizon, len(starts), starts[0].date())
            result = _run_model(model, factory, feats, horizon, starts, end)
            if not result.empty:
                results.append(result)
    if not results:
        return pd.DataFrame(columns=OUTPUT_COLS)
    out = pd.concat(results, ignore_index=True)
    out["horizon"] = out["horizon"].astype(int)
    return out.sort_values(["model", "horizon", "city", "issue_date"]).reset_index(drop=True)

"""Scores for the long backtest output (`model, city, horizon, issue_date, target_date, y, p10, p50, p90`).

Every function drops rows without a truth (`y`) or a forecast (`p50`) and reports `n`, so a table
always says how many forecasts it is based on. Comparisons between models should first go through
`paired()`: CAMS only exists from Sep 2022 and a model may skip rows, and a skill score computed on
different rows for each model is not a fair comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from thaiair import config

ROW_KEYS = ["city", "horizon", "issue_date"]  # one forecast "situation", shared by all models


def season(date: pd.Timestamp | pd.Series | pd.DatetimeIndex) -> str | pd.Series:
    """`"burning"` for months in `config.BURNING_MONTHS` (smoke season), else `"rest"`.

    Accepts a single timestamp or a Series / DatetimeIndex (returns a Series of labels).
    """
    if isinstance(date, (pd.Series, pd.DatetimeIndex)):
        months = pd.Series(pd.to_datetime(date)).dt.month
        labels = np.where(months.isin(config.BURNING_MONTHS), "burning", "rest")
        index = date.index if isinstance(date, pd.Series) else None
        return pd.Series(labels, index=index, name="season")
    return "burning" if pd.Timestamp(date).month in config.BURNING_MONTHS else "rest"


def _prepare(preds: pd.DataFrame, by: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Scored rows only (y and p50 present), with `season` derived on demand; `model` always grouped."""
    df = preds.dropna(subset=["y", "p50"])
    if "season" in by and "season" not in df.columns:
        df = df.assign(season=season(df["target_date"]).to_numpy())
    groups = ["model", *[c for c in by if c != "model"]]
    return df, groups


def paired(preds: pd.DataFrame, models: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows of `models` restricted to the forecast situations where every one of them has y and p50.

    This is the like-for-like row set: a model cannot look better just because it skipped the
    hard days (or because CAMS is missing in early years).
    """
    models = list(models) if models is not None else sorted(preds["model"].unique())
    sub = preds[preds["model"].isin(models)]
    ok = sub.dropna(subset=["y", "p50"])
    counts = ok.groupby(ROW_KEYS)["model"].nunique()
    keys = counts[counts == len(models)].index
    return ok.merge(keys.to_frame(index=False), on=ROW_KEYS)[list(preds.columns)]


def score_table(preds: pd.DataFrame, by: Sequence[str] = ("horizon",)) -> pd.DataFrame:
    """`n, mae, rmse, bias` (bias = mean of p50 - y, positive = over-forecast) per model and group."""
    df, groups = _prepare(preds, by)
    err = df["p50"] - df["y"]
    df = df.assign(abs_err=err.abs(), sq_err=err**2, err=err)
    out = df.groupby(groups).agg(
        n=("err", "size"), mae=("abs_err", "mean"), sq=("sq_err", "mean"), bias=("err", "mean")
    )
    out["rmse"] = np.sqrt(out.pop("sq"))
    return out[["n", "mae", "rmse", "bias"]].reset_index()


def skill(
    preds: pd.DataFrame, ref: str = "cams", metric: str = "mae", by: Sequence[str] = ("horizon",)
) -> pd.DataFrame:
    """`1 - metric(model) / metric(ref)` per model and group; > 0 means the model beats `ref`.

    Each model is paired with the reference on its own (the rows where both have y and p50), so
    adding a model with gaps does not shrink the comparison for the others.
    """
    if metric not in ("mae", "rmse"):
        raise ValueError("metric must be 'mae' or 'rmse'")
    if ref not in set(preds["model"]):
        raise ValueError(f"reference model {ref!r} not in predictions")
    groups = [c for c in by if c != "model"]
    cols = ["model", "ref", *groups, "n", f"{metric}_model", f"{metric}_ref", "skill"]
    rows = []
    for model in sorted(set(preds["model"]) - {ref}):
        table = score_table(paired(preds, [model, ref]), by=by).assign(_k=0)
        mine = table[table["model"] == model][[*groups, "_k", "n", metric]]
        theirs = table[table["model"] == ref][[*groups, "_k", metric]]
        both = mine.merge(theirs, on=[*groups, "_k"], suffixes=("_model", "_ref"))
        both["skill"] = 1.0 - both[f"{metric}_model"] / both[f"{metric}_ref"]
        rows.append(both.assign(model=model, ref=ref)[cols])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def alert_scores(preds: pd.DataFrame, threshold: float = config.UNHEALTHY_PM25, by: Sequence[str] = ()) -> pd.DataFrame:
    """Unhealthy-day alerts: alert if p50 > threshold, event if y > threshold.

    Returns `n, n_true_events, n_alerts, tp, precision, recall, f1` per model (and group).
    Precision is NaN when a model never alerts and recall is NaN when there were no events, rather
    than a misleading 0 or 1.
    """
    df, groups = _prepare(preds, by)
    alert, event = df["p50"] > threshold, df["y"] > threshold
    df = df.assign(alert=alert, event=event, tp=alert & event)
    out = df.groupby(groups).agg(
        n=("y", "size"), n_true_events=("event", "sum"), n_alerts=("alert", "sum"), tp=("tp", "sum")
    )
    out["precision"] = out["tp"] / out["n_alerts"].where(out["n_alerts"] > 0)
    out["recall"] = out["tp"] / out["n_true_events"].where(out["n_true_events"] > 0)
    out["f1"] = 2 * out["tp"] / (out["n_alerts"] + out["n_true_events"]).where(lambda s: s > 0)
    return out.reset_index()


def coverage(preds: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Share of y inside [p10, p90] (target 0.80) and mean interval width, per model (and group).

    Only rows with both bounds count, so models without intervals simply do not appear.
    """
    df, groups = _prepare(preds.dropna(subset=["p10", "p90"]), by)
    inside = (df["y"] >= df["p10"]) & (df["y"] <= df["p90"])
    df = df.assign(inside=inside.astype(float), width=df["p90"] - df["p10"])
    out = df.groupby(groups).agg(n=("y", "size"), coverage=("inside", "mean"), mean_width=("width", "mean"))
    return out.reset_index()

"""Turn the daily panel into a supervised table: one row per (city, issue_date) for one horizon.

Leakage rule (enforced by tests/test_features.py): a feature may use pm25 / w_* / cams_pm25 only
from dates <= issue_date, and the weather forecast f{h}_* only for target_date (that forecast was
issued at least h days earlier, so it was known at issue time). Everything below is built from
backward-looking shifts and rolling windows on a complete daily calendar, plus a single forward
lookup (shift(-h)) for the target-day values, which keeps the rule easy to audit.

Rows are emitted for every panel date, even when the target is unknown: the live app passes
future rows that carry only f* columns, and the backtest must see NaN-y rows instead of silently
losing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaiair import config, synthetic

# pm25 lags 0..6 use a forward fill of at most this many days, so one missed day (a station outage)
# does not blank out "today", while a long outage is not hidden behind a stale value.
PM_FFILL_LIMIT = 2
PM_LAGS = tuple(range(7))
PM_WINDOWS = (3, 7, 14, 30)  # rolling-mean lengths (days, ending on issue_date)
W_VARS = ("wind", "blh", "blh_min", "precip", "rh")  # observed weather with lag0 / lag1 / 3-day mean
W_SINGLE = ("u", "v", "temp")  # observed weather with lag0 only (wind direction, temperature)
FC_CHANGE_VARS = ("temp", "rh", "wind", "precip")  # forecast minus today's observed value
CATEGORICAL = ["city_code"]
ID_COLS = ["city", "issue_date", "target_date", "horizon", "y", "cams_target", "clim_doy"]

CITY_CODES: dict[str, int] = {key: i for i, key in enumerate(config.CITIES)}


def feature_columns(horizon: int) -> list[str]:
    """Model input columns for one horizon, in a fixed order (saved in the model manifest)."""
    pm = [f"pm25_lag{k}" for k in PM_LAGS] + [f"pm25_mean{w}" for w in PM_WINDOWS]
    pm += ["pm25_max7", "pm25_std7", "pm25_diff1", "pm25_days_since_obs"]
    w = [f"w_{v}_{s}" for v in W_VARS for s in ("lag0", "lag1", "mean3")] + [f"w_{v}_lag0" for v in W_SINGLE]
    cams = ["cams_lag0", "cams_lag1", "cams_mean7", "pm25_cams_ratio7"]
    fc = [f"f{horizon}_{v}" for v in synthetic.FC_VARS] + [f"f{horizon}_{v}_chg" for v in FC_CHANGE_VARS]
    cal = ["doy_sin", "doy_cos", "month", "dow", *CATEGORICAL]
    return pm + w + cams + fc + cal


def _rolling(s: pd.Series, window: int, how: str) -> pd.Series:
    """Trailing window ending on the issue date; needs about half the days observed to count."""
    return getattr(s.rolling(window, min_periods=max(2, (window + 1) // 2)), how)()


def _pm25_features(pm: pd.Series) -> dict[str, pd.Series]:
    """Recent PM2.5 level, trend and volatility: the strongest signal at 1-3 days."""
    filled = pm.ffill(limit=PM_FFILL_LIMIT)
    out = {f"pm25_lag{k}": filled.shift(k) for k in PM_LAGS}
    out |= {f"pm25_mean{w}": _rolling(pm, w, "mean") for w in PM_WINDOWS}
    out["pm25_max7"] = _rolling(pm, 7, "max")
    out["pm25_std7"] = _rolling(pm, 7, "std")
    out["pm25_diff1"] = out["pm25_lag0"] - out["pm25_lag1"]
    # How stale "today" is: 0 = observed today, 1-2 = filled, >2 = lag0 is NaN.
    pos = pd.Series(np.arange(len(pm), dtype=float), index=pm.index)
    out["pm25_days_since_obs"] = pos - pos.where(pm.notna()).ffill()
    return out


def _weather_features(g: pd.DataFrame) -> dict[str, pd.Series]:
    """Observed (ERA5) weather up to the issue date: stagnation, mixing height, wash-out by rain."""
    out: dict[str, pd.Series] = {}
    for v in W_VARS:
        s = g[f"w_{v}"]
        out[f"w_{v}_lag0"] = s
        out[f"w_{v}_lag1"] = s.shift(1)
        out[f"w_{v}_mean3"] = _rolling(s, 3, "mean")
    for v in W_SINGLE:
        out[f"w_{v}_lag0"] = g[f"w_{v}"]
    return out


def _cams_features(cams: pd.Series, pm25_mean7: pd.Series) -> dict[str, pd.Series]:
    """Past CAMS only. The ratio of observed to CAMS over the last week is a learned bias signal."""
    mean7 = _rolling(cams, 7, "mean")
    return {
        "cams_lag0": cams,
        "cams_lag1": cams.shift(1),
        "cams_mean7": mean7,
        # +1 keeps the ratio finite on very clean days without changing it on polluted ones.
        "pm25_cams_ratio7": (pm25_mean7 + 1.0) / (mean7 + 1.0),
    }


def _forecast_features(g: pd.DataFrame, horizon: int) -> dict[str, pd.Series]:
    """Weather forecast valid on the target day (issued >= h days before), and its change vs today."""
    out = {f"f{horizon}_{v}": g[f"f{horizon}_{v}"].shift(-horizon) for v in synthetic.FC_VARS}
    for v in FC_CHANGE_VARS:
        out[f"f{horizon}_{v}_chg"] = out[f"f{horizon}_{v}"] - g[f"w_{v}"]
    return out


def _calendar_features(target: pd.DatetimeIndex, city: str) -> dict[str, np.ndarray]:
    """Season of the target day (smoke season peaks Feb-Apr) and a stable city code."""
    angle = 2 * np.pi * target.day_of_year.to_numpy() / 365.25
    n = len(target)
    return {
        "doy_sin": np.sin(angle),
        "doy_cos": np.cos(angle),
        "month": target.month.to_numpy().astype(np.int64),
        "dow": target.day_of_week.to_numpy().astype(np.int64),
        "city_code": np.full(n, CITY_CODES.get(city, -1), dtype=np.int64),
    }


def _city_table(g: pd.DataFrame, city: str, horizon: int) -> pd.DataFrame:
    """Features for one city on a complete daily calendar, then cut back to the panel's own dates."""
    own_dates = pd.DatetimeIndex(g["date"])
    g = g.set_index("date").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    issue = g.index
    target = issue + pd.Timedelta(days=horizon)

    pm = _pm25_features(g["pm25"])
    cols: dict[str, object] = {
        "city": city,
        "issue_date": issue,
        "target_date": target,
        "horizon": horizon,
        "y": g["pm25"].shift(-horizon).to_numpy(),
        "cams_target": g["cams_pm25"].shift(-horizon).to_numpy(),
        "clim_doy": target.day_of_year.to_numpy().astype(np.int64),
    }
    feats = {**pm, **_weather_features(g), **_cams_features(g["cams_pm25"], pm["pm25_mean7"])}
    feats |= _forecast_features(g, horizon)
    cols |= {k: v.to_numpy(dtype=float) for k, v in feats.items()}
    cols |= _calendar_features(target, city)
    table = pd.DataFrame(cols)
    return table[table["issue_date"].isin(own_dates)]


def make_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Supervised table for one horizon: identity columns `ID_COLS` + `feature_columns(horizon)`.

    `y` and `cams_target` are pm25 / cams_pm25 on target_date (NaN when absent). Neither is a
    model feature; `cams_target` exists only for the CAMS baselines.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    if panel.duplicated(["city", "date"]).any():
        raise ValueError("panel has duplicate (city, date) rows")
    tables = [_city_table(g, str(city), horizon) for city, g in panel.groupby("city", sort=True)]
    out = pd.concat(tables, ignore_index=True)
    return out[[*ID_COLS, *feature_columns(horizon)]].reset_index(drop=True)

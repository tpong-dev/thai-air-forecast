"""Reference forecasters the models must beat. Each follows `thaiair.models.base.Forecaster`.

- `Persistence`: tomorrow = today. Hard to beat at 1 day because PM2.5 episodes last for days.
- `SeasonalAverage`: what this time of year usually looks like in this city (climatology).
- `Cams`: the raw Copernicus CAMS forecast, the operational benchmark.
- `CamsBiasCorrected`: CAMS after a per-city linear correction, i.e. "the cheap fix" a model must beat.

All four give 80% intervals the same way (empirical 10% / 90% quantiles of their training
residuals in log1p space, per city), so interval coverage can be compared with the ML models on
equal terms. Residuals are multiplicative in log space because PM2.5 errors grow with the level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaiair.models.base import PRED_COLS

MIN_ROWS = 30  # fewer training rows than this for a city -> use the pooled (all-city) estimate
SEASON_HALF_WINDOW = 7  # +/- days of year pooled by the seasonal average
SEASON_MIN_ROWS = 5  # minimum values in that window before falling back to the city-month median


def _log(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


class _ResidualIntervals:
    """Per-city empirical quantiles of log1p(y) - log1p(p50) on the training rows."""

    def __init__(self) -> None:
        self.by_city: dict[str, tuple[float, float]] = {}
        self.pooled: tuple[float, float] = (np.nan, np.nan)

    def fit(self, city: pd.Series, y: pd.Series, p50: pd.Series | np.ndarray) -> _ResidualIntervals:
        resid = pd.Series(_log(y) - _log(p50), index=city.index)
        ok = resid.notna() & pd.Series(np.isfinite(resid.to_numpy()), index=city.index)
        resid, city = resid[ok], city[ok]
        if len(resid):
            self.pooled = (float(resid.quantile(0.1)), float(resid.quantile(0.9)))
        for c, r in resid.groupby(city):
            if len(r) >= MIN_ROWS:
                self.by_city[str(c)] = (float(r.quantile(0.1)), float(r.quantile(0.9)))
        return self

    def frame(self, city: pd.Series, p50: pd.Series) -> pd.DataFrame:
        """p10 / p50 / p90 frame on the index of `city`; NaN p50 stays NaN everywhere."""
        lo = city.map(lambda c: self.by_city.get(c, self.pooled)[0]).astype(float).to_numpy()
        hi = city.map(lambda c: self.by_city.get(c, self.pooled)[1]).astype(float).to_numpy()
        mid = _log(p50)
        return pd.DataFrame(
            {"p10": np.expm1(mid + lo), "p50": np.asarray(p50, dtype=float), "p90": np.expm1(mid + hi)},
            index=city.index,
        )[PRED_COLS]


class _Baseline:
    """Shared fit/predict: subclasses only say how to make the point forecast."""

    name = "baseline"

    def __init__(self) -> None:
        self.intervals = _ResidualIntervals()

    def _fit_point(self, train: pd.DataFrame) -> None:
        """Learn whatever the point forecast needs (nothing, for most baselines)."""

    def _point(self, rows: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> _Baseline:
        self._fit_point(train)
        self.intervals = _ResidualIntervals().fit(train["city"], train["y"], self._point(train))
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        return self.intervals.frame(test["city"], self._point(test))


class Persistence(_Baseline):
    """Last observed PM2.5 on the issue date (forward-filled at most 2 days: `pm25_lag0`)."""

    name = "persistence"

    def _point(self, rows: pd.DataFrame) -> pd.Series:
        return rows["pm25_lag0"].astype(float)


class SeasonalAverage(_Baseline):
    """Median PM2.5 for the target's day of year (+/- 7 days, wrapping at New Year) in each city.

    Falls back to the city-month median, then the global median, when a window has too few values.
    """

    name = "seasonal"

    def __init__(self) -> None:
        super().__init__()
        self.table: dict[tuple[str, int], float] = {}
        self.month: dict[tuple[str, int], float] = {}
        self.overall = np.nan

    def _fit_point(self, train: pd.DataFrame) -> None:
        rows = train.dropna(subset=["y"])
        self.overall = float(rows["y"].median()) if len(rows) else np.nan
        self.month = rows.groupby(["city", rows["target_date"].dt.month])["y"].median().to_dict()
        days = np.arange(1, 367)
        self.table = {}
        for city, g in rows.groupby("city"):
            doy, y = g["clim_doy"].to_numpy(), g["y"].to_numpy()
            gap = np.abs(doy[None, :] - days[:, None])
            near = np.minimum(gap, 365 - gap) <= SEASON_HALF_WINDOW
            for d, mask in zip(days, near):
                if mask.sum() >= SEASON_MIN_ROWS:
                    self.table[(str(city), int(d))] = float(np.median(y[mask]))

    def _point(self, rows: pd.DataFrame) -> pd.Series:
        months = rows["target_date"].dt.month.to_numpy()
        values = [
            self.table.get((c, int(d)), self.month.get((c, int(m)), self.overall))
            for c, d, m in zip(rows["city"], rows["clim_doy"], months)
        ]
        return pd.Series(values, index=rows.index, dtype=float)


class Cams(_Baseline):
    """Raw CAMS daily mean on the target date. NaN where CAMS has no value (before Sep 2022)."""

    name = "cams"

    def _point(self, rows: pd.DataFrame) -> pd.Series:
        return rows["cams_target"].astype(float)


class CamsBiasCorrected(_Baseline):
    """Per-city linear fit log1p(y) = a + b * log1p(cams) on the training rows.

    Log space turns CAMS's mostly multiplicative bias (it over-predicts smoke episodes) into an
    additive one. Cities with fewer than `MIN_ROWS` CAMS rows keep raw CAMS (a=0, b=1).
    """

    name = "cams_bc"

    def __init__(self) -> None:
        super().__init__()
        self.coef: dict[str, tuple[float, float]] = {}

    def _fit_point(self, train: pd.DataFrame) -> None:
        rows = train.dropna(subset=["y", "cams_target"])
        self.coef = {}
        for city, g in rows.groupby("city"):
            x, y = _log(g["cams_target"]), _log(g["y"])
            if len(g) >= MIN_ROWS and np.ptp(x) > 0:
                b, a = np.polyfit(x, y, 1)
                self.coef[str(city)] = (float(a), float(b))

    def _point(self, rows: pd.DataFrame) -> pd.Series:
        ab = np.array([self.coef.get(c, (0.0, 1.0)) for c in rows["city"]], dtype=float).reshape(-1, 2)
        fitted = np.expm1(ab[:, 0] + ab[:, 1] * _log(rows["cams_target"]))
        fitted = np.where(rows["cams_target"].isna(), np.nan, np.clip(fitted, 0.0, None))
        return pd.Series(fitted, index=rows.index, dtype=float)


BASELINES: dict[str, type[_Baseline]] = {
    cls.name: cls for cls in (Persistence, SeasonalAverage, Cams, CamsBiasCorrected)
}

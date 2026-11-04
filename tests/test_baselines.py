"""Baselines follow the Forecaster protocol and do what their names say."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaiair import synthetic
from thaiair.baselines import BASELINES, Cams, CamsBiasCorrected, Persistence, SeasonalAverage
from thaiair.features import make_features
from thaiair.models.base import PRED_COLS, Forecaster

SPLIT = pd.Timestamp("2023-07-01")


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = make_features(synthetic.make_panel(seed=4), 1)
    train = feats[(feats["target_date"] < SPLIT) & feats["y"].notna()]
    test = feats[feats["issue_date"] >= SPLIT]
    return train, test


def _mae(pred: pd.DataFrame, test: pd.DataFrame) -> float:
    return float((pred["p50"] - test["y"]).abs().mean())


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_protocol_and_intervals(name: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = data
    model = BASELINES[name]()
    assert isinstance(model, Forecaster) and model.name == name
    pred = model.fit(train).predict(test)
    assert list(pred.columns) == PRED_COLS and pred.index.equals(test.index)
    ok = pred["p50"].notna()
    assert ok.mean() > 0.9
    assert (pred.loc[ok, "p10"] <= pred.loc[ok, "p50"]).all() and (pred.loc[ok, "p50"] <= pred.loc[ok, "p90"]).all()
    scored = ok & test["y"].notna()
    inside = (test["y"] >= pred["p10"]) & (test["y"] <= pred["p90"])
    assert 0.6 < inside[scored].mean() < 0.95  # nominal 80%


def test_names() -> None:
    assert [c.name for c in (Persistence, SeasonalAverage, Cams, CamsBiasCorrected)] == [
        "persistence",
        "seasonal",
        "cams",
        "cams_bc",
    ]


def test_persistence_is_last_observation(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = data
    pred = Persistence().fit(train).predict(test)
    np.testing.assert_array_equal(pred["p50"].to_numpy(), test["pm25_lag0"].to_numpy())


def test_seasonal_window_wraps_new_year() -> None:
    days = pd.date_range("2020-12-20", "2021-01-10")
    train = pd.DataFrame(
        {"city": "nan", "target_date": days, "clim_doy": days.day_of_year, "y": np.where(days.month == 12, 10.0, 30.0)}
    )
    model = SeasonalAverage().fit(train)
    probe = pd.DataFrame(
        {"city": ["nan", "nan", "hat_yai"], "target_date": pd.to_datetime(["2021-12-31", "2022-06-15", "2022-01-01"])}
    )
    probe["clim_doy"] = probe["target_date"].dt.day_of_year
    p50 = model.predict(probe)["p50"].to_numpy()
    assert 10.0 <= p50[0] <= 30.0  # window spans Dec and Jan values
    assert p50[1] == pytest.approx(train["y"].median())  # no data near June, no month data: global median
    assert p50[2] == pytest.approx(train["y"].median())  # unseen city: global median


def test_cams_raw_and_nan_where_missing(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = data
    test = test.copy()
    test.iloc[:20, test.columns.get_loc("cams_target")] = np.nan
    for cls in (Cams, CamsBiasCorrected):
        pred = cls().fit(train).predict(test)
        assert pred["p50"].iloc[:20].isna().all() and pred.iloc[:20].isna().all().all()
    np.testing.assert_array_equal(Cams().fit(train).predict(test)["p50"].to_numpy(), test["cams_target"].to_numpy())


def test_bias_correction_removes_multiplicative_bias() -> None:
    rng = np.random.default_rng(0)
    y = rng.gamma(3, 12, 600)
    rows = pd.DataFrame({"city": "bangkok", "y": y, "cams_target": 1.6 * y * np.exp(0.1 * rng.standard_normal(600))})
    train, test = rows.iloc[:400], rows.iloc[400:]
    raw = Cams().fit(train).predict(test)
    fixed = CamsBiasCorrected().fit(train).predict(test)
    assert _mae(fixed, test) < 0.5 * _mae(raw, test)


def test_bias_correction_falls_back_to_raw_with_little_data() -> None:
    rows = pd.DataFrame({"city": "nan", "y": [10.0, 20.0, 30.0], "cams_target": [15.0, 25.0, 50.0]})
    pred = CamsBiasCorrected().fit(rows).predict(rows)
    np.testing.assert_allclose(pred["p50"].to_numpy(), rows["cams_target"].to_numpy())

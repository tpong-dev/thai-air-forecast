"""make_features: shape, target alignment and, above all, no look-ahead leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaiair import config, synthetic
from thaiair.features import CATEGORICAL, ID_COLS, feature_columns, make_features

PAST_COLS = ["pm25", *synthetic.PAST_WEATHER_COLS, *synthetic.CAMS_COLS]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic.make_panel(start="2022-01-01", end="2022-12-31", seed=1, missing_rate=0.05)


def _garble_future(panel: pd.DataFrame, issue: pd.Timestamp, horizon: int, rng: np.random.Generator) -> pd.DataFrame:
    """Overwrite everything not known at `issue` with garbage (and some NaN).

    Kept intact: observations up to `issue`, and the h-day-old forecast f{h}_* valid on the target
    day. Everything else in the future, including other forecast columns, is replaced.
    """
    out = panel.copy()
    target = issue + pd.Timedelta(days=horizon)
    future = out["date"] > issue
    keep_fc = [f"f{horizon}_{v}" for v in synthetic.FC_VARS]
    for col in [*PAST_COLS, *synthetic.FC_WEATHER_COLS]:
        mask = future & ~((out["date"] == target) & (col in keep_fc))
        garbage = rng.uniform(-500, 5000, mask.sum())
        garbage[rng.random(mask.sum()) < 0.2] = np.nan
        out.loc[mask, col] = garbage
    return out


@pytest.mark.parametrize("horizon", config.HORIZONS)
def test_no_leakage(panel: pd.DataFrame, horizon: int) -> None:
    rng = np.random.default_rng(horizon)
    cols = feature_columns(horizon)
    base = make_features(panel, horizon).set_index(["city", "issue_date"])
    for issue in pd.to_datetime(["2022-01-03", "2022-02-15", "2022-03-31", "2022-07-01", "2022-12-28"]):
        garbled = make_features(_garble_future(panel, issue, horizon, rng), horizon).set_index(["city", "issue_date"])
        rows = base.index.get_level_values("issue_date") == issue
        pd.testing.assert_frame_equal(base.loc[rows, cols], garbled.loc[rows, cols], obj=f"features at {issue.date()}")


@pytest.mark.parametrize("horizon", config.HORIZONS)
def test_target_alignment(panel: pd.DataFrame, horizon: int) -> None:
    feats = make_features(panel, horizon)
    assert (feats["target_date"] - feats["issue_date"]).eq(pd.Timedelta(days=horizon)).all()
    truth = panel.set_index(["city", "date"])
    keys = pd.MultiIndex.from_arrays([feats["city"], feats["target_date"]])
    y = truth["pm25"].reindex(keys).to_numpy()
    cams = truth["cams_pm25"].reindex(keys).to_numpy()
    np.testing.assert_array_equal(feats["y"].to_numpy(), y)
    np.testing.assert_array_equal(feats["cams_target"].to_numpy(), cams)
    assert (feats["clim_doy"] == feats["target_date"].dt.day_of_year).all()
    assert (feats["horizon"] == horizon).all()


def test_one_row_per_panel_date_and_columns(panel: pd.DataFrame) -> None:
    feats = make_features(panel, 2)
    assert list(feats.columns) == [*ID_COLS, *feature_columns(2)]
    assert len(feats) == len(panel)
    assert not feats.duplicated(["city", "issue_date"]).any()
    assert 40 <= len(feature_columns(2)) <= 60
    assert set(CATEGORICAL) <= set(feature_columns(2))
    # Last h rows per city have no target yet, but are still present.
    last = feats.groupby("city").tail(2)
    assert last["y"].isna().all()


def test_city_code_follows_config_order(panel: pd.DataFrame) -> None:
    feats = make_features(panel, 1)
    codes = feats.groupby("city")["city_code"].first()
    order = list(config.CITIES)
    for city, code in codes.items():
        assert code == order.index(city)


def test_pm25_forward_fill_at_most_two_days() -> None:
    p = synthetic.make_panel(start="2022-01-01", end="2022-02-28", cities=["bangkok"], missing_rate=0.0)
    p.loc[10:13, "pm25"] = np.nan  # four missing days
    feats = make_features(p, 1).set_index("issue_date")
    dates = p["date"]
    last_seen = p.loc[9, "pm25"]
    assert feats.loc[dates[10], "pm25_lag0"] == last_seen
    assert feats.loc[dates[11], "pm25_lag0"] == last_seen
    assert np.isnan(feats.loc[dates[12], "pm25_lag0"])
    assert feats.loc[dates[12], "pm25_days_since_obs"] == 3
    assert feats.loc[dates[14], "pm25_days_since_obs"] == 0


def test_future_rows_with_only_forecasts(panel: pd.DataFrame) -> None:
    """The live app appends rows for the next days that carry only f* columns."""
    city = panel[panel["city"] == "bangkok"]
    last = city["date"].max()
    future = pd.DataFrame({"city": "bangkok", "date": pd.date_range(last + pd.Timedelta(days=1), periods=3)})
    for col in synthetic.FC_WEATHER_COLS:
        future[col] = 7.0
    live = pd.concat([city, future], ignore_index=True)[synthetic.PANEL_COLS]
    feats = make_features(live, 3).set_index("issue_date")
    row = feats.loc[last]
    assert np.isnan(row["y"])
    assert row["f3_wind"] == 7.0
    assert row["pm25_lag0"] == pytest.approx(city["pm25"].ffill(limit=2).iloc[-1], nan_ok=True)

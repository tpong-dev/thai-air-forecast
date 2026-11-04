"""Tests for thaiair.weather: aggregation rules, wind components, contract columns, live frame.

No network and no data files: Open-Meteo responses are built in memory and HTTP is mocked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from thaiair import http, synthetic, weather


def make_payload(start: str, days: int, series: dict[str, Any], offset: int = 25200) -> dict[str, Any]:
    """A minimal Open-Meteo-style JSON payload with hourly local times from ``start``."""
    times = pd.date_range(start, periods=24 * days, freq="h")
    hourly: dict[str, Any] = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for key, value in series.items():
        values = value(times) if callable(value) else [value] * len(times)
        hourly[key] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in values]
    return {"utc_offset_seconds": offset, "timezone": "Asia/Bangkok", "hourly": hourly}


class FakeResponse:
    def __init__(self, data: Any) -> None:
        self._data = data

    def json(self) -> Any:
        return self._data


def hourly(days: int = 2, **cols: Any) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", name="time")
    return pd.DataFrame({k: (v if np.ndim(v) else np.full(len(idx), v, dtype=float)) for k, v in cols.items()}, index=idx)


# ---------------------------------------------------------------- wind components


@pytest.mark.parametrize(
    ("direction", "u", "v"),
    [(0, 0.0, -2.0), (90, -2.0, 0.0), (180, 0.0, 2.0), (270, 2.0, 0.0)],
)
def test_wind_to_uv_uses_from_direction(direction: float, u: float, v: float) -> None:
    # A wind *from* the north (0 deg) blows towards the south, so v is negative.
    got_u, got_v = weather.wind_to_uv(np.array([2.0]), np.array([direction]))
    assert got_u[0] == pytest.approx(u, abs=1e-9)
    assert got_v[0] == pytest.approx(v, abs=1e-9)


def test_daily_uv_avoids_angle_wraparound() -> None:
    # Half the day from 350 deg, half from 10 deg: the mean wind is from the north, not from the south.
    wdir = np.r_[np.full(12, 350.0), np.full(12, 10.0)]
    day = weather.daily_aggregate(hourly(1, wind=2.0, wdir=wdir))
    assert day["u"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert day["v"].iloc[0] < -1.9


# ---------------------------------------------------------------- daily aggregation


def test_daily_aggregate_means_sums_and_min() -> None:
    temp = np.r_[np.arange(24.0), np.full(24, 30.0)]
    blh = np.r_[np.linspace(100, 1250, 24), np.full(24, 500.0)]
    df = weather.daily_aggregate(hourly(2, temp=temp, rh=70.0, wind=3.0, wdir=270.0, precip=0.5, blh=blh))
    assert list(df.columns) == ["temp", "rh", "wind", "u", "v", "precip", "blh", "blh_min"]
    assert df.index.name == "date"
    assert df["temp"].tolist() == pytest.approx([11.5, 30.0])
    assert df["precip"].tolist() == pytest.approx([12.0, 12.0])  # 0.5 mm/h * 24 h
    assert df["blh_min"].iloc[0] == pytest.approx(100.0)
    assert df["blh"].iloc[0] == pytest.approx(675.0)
    assert df["u"].iloc[0] == pytest.approx(3.0)  # from the west -> towards the east


def test_min_hours_rule() -> None:
    temp = np.full(48, 25.0)
    temp[:5] = np.nan  # day 1: 19 valid hours -> NaN
    temp[24:28] = np.nan  # day 2: 20 valid hours -> kept
    df = weather.daily_aggregate(hourly(2, temp=temp))
    assert np.isnan(df["temp"].iloc[0])
    assert df["temp"].iloc[1] == pytest.approx(25.0)


def test_precip_with_missing_hours_is_not_undercounted() -> None:
    precip = np.full(24, 1.0)
    precip[:3] = np.nan  # 21 valid hours, still a valid day
    df = weather.daily_aggregate(hourly(1, precip=precip))
    assert df["precip"].iloc[0] == pytest.approx(24.0)


def test_six_hour_blocks_spread_evenly_sum_correctly() -> None:
    # JMA precipitation comes as 6 h totals spread evenly over the hours (e.g. 3.0 x 6 = 18 mm).
    block_totals = [0.0, 18.0, 6.0, 0.0]
    precip = np.repeat(np.array(block_totals) / 6, 6)
    df = weather.daily_aggregate(hourly(1, precip=precip))
    assert df["precip"].iloc[0] == pytest.approx(24.0)


def test_three_hourly_series_with_gaps() -> None:
    # Values only every 3rd hour, nulls in between: 8 values a day count as a full day.
    temp = np.full(48, np.nan)
    precip = np.full(48, np.nan)
    temp[::3] = 20.0
    precip[::3] = 0.5  # mm/h rate at each 3-hourly step
    temp[24 + 0] = temp[24 + 3] = np.nan  # day 2: 6 of 8 values < 7 needed -> NaN
    h = hourly(2, temp=temp, precip=precip)
    assert weather.native_step_hours(h["temp"]) == 3
    df = weather.daily_aggregate(h)
    assert df["temp"].iloc[0] == pytest.approx(20.0)
    assert np.isnan(df["temp"].iloc[1])
    assert df["precip"].tolist() == pytest.approx([12.0, 12.0])  # not 8 * 0.5 = 4 mm


def test_native_step_defaults_to_hourly() -> None:
    assert weather.native_step_hours(pd.Series([np.nan, np.nan], index=pd.date_range("2024", periods=2, freq="h"))) == 1
    assert weather.native_step_hours(hourly(1, temp=1.0)["temp"]) == 1


# ---------------------------------------------------------------- parsing and chunking


def test_hourly_frame_checks_timezone_and_fills_missing_keys() -> None:
    payload = make_payload("2024-01-01", 1, {"temperature_2m": 25.0})
    df = weather.hourly_frame(payload, {"temperature_2m": "temp", "boundary_layer_height": "blh"})
    assert df["temp"].eq(25.0).all()
    assert df["blh"].isna().all()
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00")
    with pytest.raises(ValueError, match="utc_offset_seconds"):
        weather.hourly_frame(make_payload("2024-01-01", 1, {"temperature_2m": 25.0}, offset=0), {"temperature_2m": "t"})


def test_date_chunks() -> None:
    chunks = weather.date_chunks("2019-01-01", "2020-03-15", 12)
    assert chunks == [
        (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-15")),
    ]
    q = weather.date_chunks("2022-07-01", "2022-12-31", 3)
    assert [c[1] for c in q] == [pd.Timestamp("2022-09-30"), pd.Timestamp("2022-12-31")]


def test_complete_calendar_fills_gaps_and_trims_trailing_nan() -> None:
    df = pd.DataFrame(
        {
            "city": ["b", "b", "b", "a"],
            "date": pd.to_datetime(["2024-01-03", "2024-01-05", "2024-01-06", "2024-01-02"]),
            "x": [1.0, 2.0, np.nan, 3.0],
        }
    )
    out = weather.complete_calendar(df, ["x"], "2024-01-01")
    assert out["city"].tolist() == ["a", "a", "b", "b", "b", "b", "b"]
    b = out[out.city == "b"]
    assert b["date"].max() == pd.Timestamp("2024-01-05")  # trailing NaN day dropped
    assert b["x"].isna().sum() == 3  # 01, 02, 04
    assert out["date"].dtype == "datetime64[ns]"


# ---------------------------------------------------------------- builders with mocked HTTP


def _era5_payload(start: str, days: int) -> dict[str, Any]:
    return make_payload(
        start,
        days,
        {
            "temperature_2m": 28.0,
            "relative_humidity_2m": 70.0,
            "wind_speed_10m": 2.0,
            "wind_direction_10m": 0.0,
            "precipitation": 0.1,
            "boundary_layer_height": lambda t: 200 + 10 * t.hour.to_numpy(),
        },
    )


def _jma_payload(start: str, days: int) -> dict[str, Any]:
    series: dict[str, Any] = {}
    for h in (1, 2, 3):
        series |= {
            f"temperature_2m_previous_day{h}": 27.0 + h,
            f"relative_humidity_2m_previous_day{h}": 60.0,
            f"wind_speed_10m_previous_day{h}": 1.0,
            f"wind_direction_10m_previous_day{h}": 90.0,
            f"precipitation_previous_day{h}": 0.0,
        }
    return make_payload(start, days, series)


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> list[dict[str, Any]]:
    """Replace the network with payload builders; returns the list of requests made."""
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any] | None = None, **_: Any) -> FakeResponse:
        params = dict(params or {})
        calls.append({"url": url, **params})
        start, end = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
        days = (end - start).days + 1
        builder = _era5_payload if "archive" in url else _jma_payload
        return FakeResponse(builder(params["start_date"], days))

    monkeypatch.setattr(http, "get", fake_get)
    monkeypatch.setattr(weather, "ERA5_CACHE", tmp_path / "era5")
    monkeypatch.setattr(weather, "JMA_CACHE", tmp_path / "jma")
    monkeypatch.setattr(weather, "PAUSE_S", 0.0)
    return calls


def test_build_weather_past_contract_and_cache(fake_http: list[dict[str, Any]]) -> None:
    df = weather.build_weather_past(["bangkok"], start="2023-12-30", end="2024-01-02")
    assert list(df.columns) == ["city", "date", *synthetic.PAST_WEATHER_COLS]
    assert len(df) == 4 and df["date"].is_monotonic_increasing
    assert df["w_blh_min"].eq(200.0).all()
    assert df["w_v"].iloc[0] == pytest.approx(-2.0)  # north wind
    assert len(fake_http) == 1  # 4 days fit in one 12-month chunk
    assert fake_http[0]["timezone"] == "Asia/Bangkok" and fake_http[0]["wind_speed_unit"] == "ms"
    assert fake_http[0]["models"] == "era5"
    weather.build_weather_past(["bangkok"], start="2023-12-30", end="2024-01-02")
    assert len(fake_http) == 1  # second build is served from the cache


def test_build_weather_fc_contract(fake_http: list[dict[str, Any]]) -> None:
    df = weather.build_weather_fc(["chiang_mai"], start="2024-01-01", end="2024-01-03")
    assert list(df.columns) == ["city", "date", *synthetic.FC_WEATHER_COLS]
    assert df["f1_temp"].eq(28.0).all() and df["f3_temp"].eq(30.0).all()
    assert df["f2_u"].iloc[0] == pytest.approx(-1.0)  # from the east
    assert fake_http[0]["models"] == "jma_gsm"
    assert "temperature_2m_previous_day3" in fake_http[0]["hourly"]
    assert "boundary_layer_height" not in fake_http[0]["hourly"]


def test_cached_json_retries_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[Any] = [ValueError("empty body"), {"ok": 1}]

    def fake_get_json(*_: Any, **__: Any) -> Any:
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(http, "get_json", fake_get_json)
    monkeypatch.setattr(weather.time, "sleep", lambda _s: None)
    assert weather.cached_json("https://x", {}, cache=None) == {"ok": 1}


# ---------------------------------------------------------------- live


def _live_payload(today: str, past_days: int) -> dict[str, Any]:
    start = (pd.Timestamp(today) - pd.Timedelta(days=past_days)).strftime("%Y-%m-%d")
    days = past_days + 4
    return make_payload(
        start,
        days,
        {
            "temperature_2m_jma_gsm": lambda t: 25.0 + (t.normalize() - t[0].normalize()).days.to_numpy(),
            "relative_humidity_2m_jma_gsm": 65.0,
            "wind_speed_10m_jma_gsm": 3.0,
            "wind_direction_10m_jma_gsm": 180.0,
            "precipitation_jma_gsm": 0.25,
            "boundary_layer_height_jma_gsm": None,
            "boundary_layer_height_best_match": 800.0,
            "temperature_2m_best_match": 99.0,  # must be ignored: training uses the JMA family
        },
    )


def test_live_daily_with_mocked_json(monkeypatch: pytest.MonkeyPatch) -> None:
    today = pd.Timestamp("2026-11-04")
    seen: dict[str, Any] = {}

    def fake_get_json(url: str, params: dict[str, Any] | None = None, **_: Any) -> Any:
        seen.update(params or {}, url=url)
        return _live_payload("2026-11-04", 5)

    monkeypatch.setattr(http, "get_json", fake_get_json)
    df = weather.live_daily("nan", past_days=5, today=today)

    assert seen["url"] == weather.FORECAST_URL and seen["models"] == "jma_gsm,best_match"
    assert seen["past_days"] == 5 and seen["forecast_days"] == 4
    assert list(df.columns) == ["city", "date", *synthetic.PAST_WEATHER_COLS, *synthetic.FC_WEATHER_COLS]
    assert df["date"].tolist() == list(pd.date_range("2026-10-30", "2026-11-07"))
    assert (df["city"] == "nan").all()
    past = df["date"] < today
    assert df.loc[past, "w_temp"].tolist() == pytest.approx([25, 26, 27, 28, 29])
    assert df.loc[past, "w_blh"].eq(800.0).all()  # filled from best_match
    assert df.loc[past, "w_precip"].eq(6.0).all()
    assert df.loc[~past, synthetic.PAST_WEATHER_COLS].isna().all().all()
    assert df.loc[past, synthetic.FC_WEATHER_COLS].isna().all().all()
    fut = df[~past]
    assert fut["f1_temp"].tolist() == pytest.approx([30, 31, 32, 33])
    for v in synthetic.FC_VARS:
        assert fut[f"f1_{v}"].equals(fut[f"f2_{v}"]) and fut[f"f1_{v}"].equals(fut[f"f3_{v}"])
    assert fut["f1_v"].iloc[0] == pytest.approx(3.0)  # from the south -> towards the north

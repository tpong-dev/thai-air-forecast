"""Tests for thaiair.cams: daily mean rule, chunked build with cache, live frame. No network."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from thaiair import cams, http


def make_payload(start: str, days: int, values: Any, offset: int = 25200) -> dict[str, Any]:
    times = pd.date_range(start, periods=24 * days, freq="h")
    vals = values(times) if callable(values) else [values] * len(times)
    return {
        "utc_offset_seconds": offset,
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "pm2_5": [None if v is None or np.isnan(v) else float(v) for v in vals],
        },
    }


class FakeResponse:
    def __init__(self, data: Any) -> None:
        self._data = data

    def json(self) -> Any:
        return self._data


def test_daily_cams_mean_and_min_hours() -> None:
    idx = pd.date_range("2022-08-04", periods=48, freq="h", name="time")
    pm = np.r_[np.full(24, np.nan), np.full(24, 20.0)]
    pm[7:24] = 10.0  # first day: only 17 hours, like the real archive start (07:00 onwards)
    daily = cams.daily_cams(pd.DataFrame({"pm25": pm}, index=idx))
    assert daily.name == "cams_pm25" and daily.index.name == "date"
    assert np.isnan(daily.iloc[0])
    assert daily.iloc[1] == pytest.approx(20.0)


def test_build_cams_daily_contract_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any] | None = None, **_: Any) -> FakeResponse:
        params = dict(params or {})
        calls.append({"url": url, **params})
        days = (pd.Timestamp(params["end_date"]) - pd.Timestamp(params["start_date"])).days + 1
        # Null before 2022-08-05, then a value that depends on the day of month.
        return FakeResponse(
            make_payload(
                params["start_date"],
                days,
                lambda t: np.where(t < pd.Timestamp("2022-08-05"), np.nan, t.day.to_numpy().astype(float)),
            )
        )

    monkeypatch.setattr(http, "get", fake_get)
    monkeypatch.setattr(cams, "CAMS_CACHE", tmp_path / "cams")
    monkeypatch.setattr("thaiair.weather.PAUSE_S", 0.0)

    df = cams.build_cams_daily(["bangkok", "hat_yai"], start="2022-07-01", end="2022-10-10")
    assert list(df.columns) == ["city", "date", "cams_pm25"]
    assert not df.duplicated(["city", "date"]).any()
    assert df.groupby("city")["date"].agg(["min", "max"]).eq(
        [pd.Timestamp("2022-07-01"), pd.Timestamp("2022-10-10")]
    ).all().all()
    first = df.dropna().groupby("city")["date"].min()
    assert (first == pd.Timestamp("2022-08-05")).all()
    assert df.loc[(df.city == "bangkok") & (df.date == "2022-09-17"), "cams_pm25"].item() == pytest.approx(17.0)
    assert len(calls) == 4  # 2 cities x 2 three-month chunks
    assert calls[0]["domains"] == "cams_global" and calls[0]["hourly"] == "pm2_5"
    assert "wind_speed_unit" not in calls[0] and calls[0]["timezone"] == "Asia/Bangkok"
    cams.build_cams_daily(["bangkok", "hat_yai"], start="2022-07-01", end="2022-10-10")
    assert len(calls) == 4  # cached


def test_live_daily_with_mocked_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_get_json(url: str, params: dict[str, Any] | None = None, **_: Any) -> Any:
        seen.update(params or {}, url=url)
        return make_payload("2026-10-25", 14, 30.0)

    monkeypatch.setattr(http, "get_json", fake_get_json)
    df = cams.live_daily("chiang_rai", past_days=10, today=pd.Timestamp("2026-11-04"))
    assert seen["past_days"] == 10 and seen["forecast_days"] == 4 and seen["domains"] == "cams_global"
    assert list(df.columns) == ["city", "date", "cams_pm25"]
    assert df["date"].tolist() == list(pd.date_range("2026-10-25", "2026-11-07"))
    assert df["cams_pm25"].eq(30.0).all()
    assert df["date"].dtype == "datetime64[ns]"

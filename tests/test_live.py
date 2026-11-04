"""thaiair.live with monkeypatched fetchers: synthetic rows in, forecast dict out, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from thaiair import config, live, synthetic

TODAY = pd.Timestamp("2024-03-10")  # inside the synthetic burning season
CITY = "chiang_mai"


def _synthetic_sources(city: str = CITY) -> dict[str, pd.DataFrame]:
    """Split a synthetic panel the way the three live sources return it."""
    panel = synthetic.make_panel("2023-12-01", "2024-03-20", cities=[city], missing_rate=0.0)
    past = panel[panel["date"] < TODAY]
    return {
        "pm25": past[["city", "date", "pm25", "n_stations"]],
        "weather": panel[["city", "date", *synthetic.PAST_WEATHER_COLS, *synthetic.FC_WEATHER_COLS]],
        "cams": panel[["city", "date", "cams_pm25"]],
    }


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    """Patch the fetchers and make every saved-model load fail (empty models dir)."""
    src = _synthetic_sources()
    monkeypatch.setattr(live, "_fetch_pm25", lambda city, days: src["pm25"])
    monkeypatch.setattr(live, "_fetch_weather", lambda city, past_days: src["weather"])
    monkeypatch.setattr(live, "_fetch_cams", lambda city, past_days: src["cams"])

    def no_model(name: str, horizon: int):
        raise FileNotFoundError(f"no {name}_h{horizon}")

    monkeypatch.setattr(live, "_load_model", no_model)
    live.clear_cache()
    yield src
    live.clear_cache()


def test_build_live_panel_shape(sources) -> None:
    panel = live.build_live_panel(CITY, today=TODAY)
    assert list(panel.columns) == synthetic.PANEL_COLS
    assert panel["date"].min() == TODAY - pd.Timedelta(days=live.PAST_DAYS)
    assert panel["date"].max() == TODAY + pd.Timedelta(days=3)
    assert panel["date"].is_unique and panel["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert panel.loc[panel["date"] >= TODAY, "pm25"].isna().all()  # not observed yet
    assert panel.loc[panel["date"] == TODAY + pd.Timedelta(days=2), "cams_pm25"].notna().all()
    assert panel.attrs["warnings"] == []


def test_forecast_persistence_fallback_structure(sources) -> None:
    fc = live.forecast(CITY, today=TODAY)
    assert fc["city"] == CITY and fc["name"] == "Chiang Mai"
    assert fc["issue_date"] == "2024-03-09"
    assert fc["model"] == live.FALLBACK_LABEL
    assert [f["horizon"] for f in fc["forecasts"]] == list(config.HORIZONS)
    assert [f["date"] for f in fc["forecasts"]] == ["2024-03-10", "2024-03-11", "2024-03-12"]

    last_obs = round(float(sources["pm25"]["pm25"].iloc[-1]), 1)
    for f in fc["forecasts"]:
        assert f["p50"] == pytest.approx(last_obs, abs=0.11)  # persistence = last observed day
        assert f["p10"] <= f["p50"] <= f["p90"]
        assert f["aqi_band"] == config.thai_aqi_band(f["p50"])
        assert f["unhealthy"] == (f["p50"] > config.UNHEALTHY_PM25)
        assert f["cams"] is not None
    assert len(fc["recent"]) == live.RECENT_DAYS
    assert fc["recent"][-1]["date"] == "2024-03-09"
    assert "Copernicus Atmosphere Monitoring Service" in fc["attribution"]
    json.dumps(fc)  # JSON-serialisable (no NaN, no Timestamps)


def test_weather_and_cams_failure_degrade(sources, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(live, "_fetch_weather", boom)
    monkeypatch.setattr(live, "_fetch_cams", boom)
    fc = live.forecast(CITY, today=TODAY)
    assert len(fc["warnings"]) == 2
    assert all(f["cams"] is None and f["p50"] is not None for f in fc["forecasts"])


def test_pm25_failure_raises_clear_error(sources, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_fetch_pm25", lambda city, days: sources["pm25"].iloc[0:0])
    with pytest.raises(live.LiveDataError, match="no PM2.5"):
        live.forecast(CITY, today=TODAY)


def test_unknown_city() -> None:
    with pytest.raises(live.UnknownCityError, match="valid keys"):
        live.forecast("atlantis")


def test_cache_hits_within_ttl(sources, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(live, "_fetch_pm25", lambda city, days: calls.append(city) or sources["pm25"])
    first = live.forecast(CITY, today=TODAY)
    assert live.forecast(CITY, today=TODAY) is first
    assert len(calls) == 1
    live.forecast(CITY, today=TODAY, use_cache=False)
    assert len(calls) == 2


def test_cli_writes_partial_results(sources, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def pm25(city: str, days: int) -> pd.DataFrame:
        if city != CITY:
            raise ConnectionError("Air4Thai down")
        return sources["pm25"]

    monkeypatch.setattr(live, "_fetch_pm25", pm25)
    monkeypatch.setattr(live, "bangkok_today", lambda: TODAY)
    out = tmp_path / "live.json"
    assert live.main(["--all", "--out", str(out)]) == 0
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [r["city"] for r in results] == list(config.CITIES)
    assert sum("error" not in r for r in results) == 1


def test_end_to_end_with_tiny_lgbm(sources, monkeypatch: pytest.MonkeyPatch) -> None:
    lgbm = pytest.importorskip("thaiair.models.lgbm")
    features = pytest.importorskip("thaiair.features")
    panel = synthetic.make_panel("2021-01-01", "2023-12-31", cities=[CITY])
    fitted = {}
    for h in config.HORIZONS:
        train = features.make_features(panel, h).dropna(subset=["y"])
        model = lgbm.LightGBMQuantile(h, params={"n_estimators": 30, "num_leaves": 7})
        fitted[h] = model.fit(train)
    monkeypatch.setattr(live, "_load_model", lambda name, h: fitted[h])

    fc = live.forecast(CITY, today=TODAY)
    assert fc["model"] == live.SERVED_MODEL
    p50 = np.array([f["p50"] for f in fc["forecasts"]])
    assert np.isfinite(p50).all() and (p50 >= 0).all()
    assert all(f["p10"] <= f["p50"] <= f["p90"] for f in fc["forecasts"])

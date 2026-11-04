"""FastAPI endpoints with injected providers: no network, no model or report files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import api
from thaiair import config, live


def _fake_forecast(city: str) -> dict[str, Any]:
    c = config.CITIES[city]
    return {
        "city": city,
        "name": c.name,
        "lat": c.lat,
        "lon": c.lon,
        "issue_date": "2024-03-09",
        "generated_at": "2024-03-10T08:30:00+07:00",
        "model": live.FALLBACK_LABEL,
        "forecasts": [
            {
                "date": f"2024-03-1{h - 1}",
                "horizon": h,
                "p10": 40.0,
                "p50": 50.0,
                "p90": 70.0,
                "aqi_band": config.thai_aqi_band(50.0),
                "unhealthy": True,
                "cams": 61.2,
                "model": live.FALLBACK_LABEL,
            }
            for h in config.HORIZONS
        ],
        "recent": [{"date": "2024-03-09", "pm25": 48.3}, {"date": "2024-03-08", "pm25": None}],
        "warnings": [],
        "attribution": live.ATTRIBUTION,
    }


def _backtest_file(path: Path) -> Path:
    """Small long-format backtest: lgbm (error 2), cams (error 5), persistence, for 10 days."""
    dates = pd.date_range("2024-01-01", periods=10)
    rows = []
    for model, err in [("lgbm", 2.0), ("cams", 5.0), ("persistence", 3.0)]:
        for d in dates:
            y = 40.0 + d.day
            rows.append(
                {
                    "model": model,
                    "city": "chiang_mai",
                    "horizon": 1,
                    "issue_date": d - pd.Timedelta(days=1),
                    "target_date": d,
                    "y": y,
                    "p10": np.nan if model == "cams" else y - 10,
                    "p50": y + err,
                    "p90": np.nan if model == "cams" else y + 10,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    api.app.dependency_overrides[api.get_forecast_fn] = lambda: _fake_forecast
    api.app.dependency_overrides[api.get_backtest_path] = lambda: _backtest_file(tmp_path / "bt.parquet")
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "available_models", lambda: [])
    body = client.get("/health").json()
    assert body == {"status": "ok", "serving": live.FALLBACK_LABEL, "models": []}

    monkeypatch.setattr(live, "available_models", lambda: [f"lgbm_h{h}" for h in config.HORIZONS])
    assert client.get("/health").json()["serving"] == "lgbm"


def test_cities(client: TestClient) -> None:
    body = client.get("/cities").json()
    assert [c["key"] for c in body] == list(config.CITIES)
    assert body[1]["stations"] == ["35T", "36T"]


def test_forecast_ok(client: TestClient) -> None:
    resp = client.get("/forecast", params={"city": "chiang_mai"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["city"] == "chiang_mai" and len(body["forecasts"]) == 3
    assert body["forecasts"][0]["aqi_band"] == "starts to affect health"
    assert body["recent"][1]["pm25"] is None


def test_forecast_unknown_city_lists_valid_keys(client: TestClient) -> None:
    resp = client.get("/forecast", params={"city": "atlantis"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["valid"] == list(config.CITIES)


def test_forecast_upstream_failure_is_503(client: TestClient) -> None:
    def failing(city: str) -> dict[str, Any]:
        raise live.LiveDataError("Air4Thai returned no PM2.5 observations")

    api.app.dependency_overrides[api.get_forecast_fn] = lambda: failing
    resp = client.get("/forecast", params={"city": "bangkok"})
    assert resp.status_code == 503
    assert "Air4Thai" in resp.json()["detail"]


def test_backtest_series_and_mae(client: TestClient) -> None:
    resp = client.get(
        "/backtest", params={"city": "chiang_mai", "horizon": 1, "start": "2024-01-03", "end": "2024-01-07"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "lgbm"
    assert body["models_available"] == ["lgbm", "persistence"]
    assert [p["target_date"] for p in body["series"]] == [f"2024-01-0{d}" for d in range(3, 8)]
    assert body["n"] == 5
    assert body["mae_model"] == pytest.approx(2.0) and body["mae_cams"] == pytest.approx(5.0)
    first = body["series"][0]
    assert first["actual"] == 43.0 and first["p50"] == 45.0 and first["cams"] == 48.0


def test_backtest_other_model_and_errors(client: TestClient, tmp_path: Path) -> None:
    body = client.get("/backtest", params={"city": "chiang_mai", "model": "persistence"}).json()
    assert body["mae_model"] == pytest.approx(3.0) and len(body["series"]) == 10

    assert client.get("/backtest", params={"city": "chiang_mai", "model": "nope"}).status_code == 404
    assert client.get("/backtest", params={"city": "atlantis"}).status_code == 404
    assert client.get("/backtest", params={"city": "chiang_mai", "horizon": 9}).status_code == 422

    api.app.dependency_overrides[api.get_backtest_path] = lambda: tmp_path / "missing.parquet"
    resp = client.get("/backtest", params={"city": "chiang_mai"})
    assert resp.status_code == 404
    assert "thaiair.evaluate" in resp.json()["detail"]


def test_backtest_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("THAIAIR_BACKTEST_PATH", str(tmp_path / "x.parquet"))
    assert api.get_backtest_path() == tmp_path / "x.parquet"

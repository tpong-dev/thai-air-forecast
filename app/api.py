"""FastAPI service for the Thai PM2.5 forecast.

    uvicorn app.api:app --port 8000        (with PYTHONPATH=src;.)

Endpoints:
    GET /health                       status and which saved models were found
    GET /cities                       the 7 cities with coordinates and Air4Thai stations
    GET /forecast?city=chiang_mai     day +1/+2/+3 forecast (p10/p50/p90, Thai AQI band, CAMS)
    GET /backtest?city=&horizon=&start=&end=&model=
                                      actual vs model vs CAMS from reports/backtest_predictions.parquet

The forecast and backtest providers are FastAPI dependencies, so tests swap them with
`app.dependency_overrides` and never touch the network or the data files.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from thaiair import config, live

log = logging.getLogger(__name__)

CAMS_MODEL = "cams"  # name of the CAMS baseline in the backtest file
DEFAULT_BACKTEST_MODEL = os.environ.get("THAIAIR_SERVED_MODEL", "lgbm")
BACKTEST_COLS = ["model", "city", "horizon", "issue_date", "target_date", "y", "p10", "p50", "p90"]

ForecastFn = Callable[[str], dict[str, Any]]


# --------------------------------------------------------------------------- schemas


class HealthResponse(BaseModel):
    status: Literal["ok"]
    serving: str  # the model /forecast will use, or the fallback label
    models: list[str]  # saved models found, e.g. ["lgbm_h1", "lgbm_h2", "lgbm_h3"]


class CityOut(BaseModel):
    key: str
    name: str
    lat: float
    lon: float
    stations: list[str]


class ForecastDay(BaseModel):
    date: str
    horizon: int
    p10: float | None
    p50: float | None
    p90: float | None
    aqi_band: str | None
    unhealthy: bool
    cams: float | None
    model: str


class ObservedDay(BaseModel):
    date: str
    pm25: float | None


class ForecastResponse(BaseModel):
    city: str
    name: str
    lat: float
    lon: float
    issue_date: str
    generated_at: str
    model: str
    forecasts: list[ForecastDay]
    recent: list[ObservedDay]
    warnings: list[str] = []
    attribution: str


class BacktestPoint(BaseModel):
    issue_date: str
    target_date: str
    actual: float | None
    p10: float | None
    p50: float | None
    p90: float | None
    cams: float | None


class BacktestResponse(BaseModel):
    city: str
    horizon: int
    model: str
    models_available: list[str]
    start: str | None
    end: str | None
    n: int  # days where actual, model and CAMS all exist (the rows both MAEs use)
    mae_model: float | None
    mae_cams: float | None
    series: list[BacktestPoint]


# --------------------------------------------------------------------------- providers (injectable)


def get_forecast_fn() -> ForecastFn:
    """The live forecaster (`thaiair.live` imports its data modules lazily, so the API starts even if one is broken)."""
    return live.forecast


def get_backtest_path() -> Path:
    """Backtest predictions file; `THAIAIR_BACKTEST_PATH` overrides it (tests, Docker volumes)."""
    default = config.REPORTS_DIR / "backtest_predictions.parquet"
    return Path(os.environ.get("THAIAIR_BACKTEST_PATH", default))


# --------------------------------------------------------------------------- backtest helpers (shared with the dashboard)


@lru_cache(maxsize=4)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:
    """Cached by (path, mtime) so a re-run backtest is picked up without restarting the server."""
    df = pd.read_parquet(path, columns=BACKTEST_COLS)
    for col in ("issue_date", "target_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def load_backtest(path: Path) -> pd.DataFrame:
    """Read the long-format backtest predictions. Raises FileNotFoundError if the backtest has not run."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run `python -m thaiair.evaluate` first")
    return _read_parquet(str(path), path.stat().st_mtime)


def _mae(err: pd.Series) -> float | None:
    return round(float(err.abs().mean()), 2) if len(err) else None


def backtest_view(
    preds: pd.DataFrame,
    city: str,
    horizon: int,
    start: date | None = None,
    end: date | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Actual vs one model vs CAMS for a city/horizon/target-date window, plus MAE on common days.

    Both MAEs are computed on exactly the same days (actual, model and CAMS all present); otherwise
    a model could look better just by skipping hard days.
    """
    sel = preds[(preds["city"] == city) & (preds["horizon"] == horizon)]
    if start is not None:
        sel = sel[sel["target_date"] >= pd.Timestamp(start)]
    if end is not None:
        sel = sel[sel["target_date"] <= pd.Timestamp(end)]
    models = sorted(m for m in preds["model"].unique() if m != CAMS_MODEL)
    if model is None:
        model = DEFAULT_BACKTEST_MODEL if DEFAULT_BACKTEST_MODEL in models else (models[0] if models else "")
    if model not in models:
        raise LookupError(f"model {model!r} not in backtest; available: {', '.join(models) or 'none'}")

    keys = ["issue_date", "target_date"]
    mod = sel[sel["model"] == model][[*keys, "y", "p10", "p50", "p90"]]
    cams = sel[sel["model"] == CAMS_MODEL][[*keys, "p50"]].rename(columns={"p50": "cams"})
    merged = mod.merge(cams, on=keys, how="left").sort_values("target_date")
    common = merged.dropna(subset=["y", "p50", "cams"])

    def num(x: Any) -> float | None:
        return None if pd.isna(x) else round(float(x), 2)

    series = [
        {
            "issue_date": r.issue_date.date().isoformat(),
            "target_date": r.target_date.date().isoformat(),
            "actual": num(r.y),
            "p10": num(r.p10),
            "p50": num(r.p50),
            "p90": num(r.p90),
            "cams": num(r.cams),
        }
        for r in merged.itertuples(index=False)
    ]
    return {
        "city": city,
        "horizon": horizon,
        "model": model,
        "models_available": models,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "n": len(common),
        "mae_model": _mae(common["p50"] - common["y"]),
        "mae_cams": _mae(common["cams"] - common["y"]),
        "series": series,
    }


# --------------------------------------------------------------------------- app

app = FastAPI(
    title="Thai Air Forecast",
    version="0.1.0",
    description="Daily mean PM2.5 for 7 Thai cities, 1-3 days ahead, compared with the CAMS forecast.",
)


def _unknown_city(city: str) -> HTTPException:
    return HTTPException(404, {"error": f"unknown city {city!r}", "valid": list(config.CITIES)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = live.available_models()
    served = [m for m in models if m.startswith(f"{live.SERVED_MODEL}_h")]
    serving = live.SERVED_MODEL if len(served) == len(config.HORIZONS) else live.FALLBACK_LABEL
    return HealthResponse(status="ok", serving=serving, models=models)


@app.get("/cities", response_model=list[CityOut])
def cities() -> list[CityOut]:
    return [
        CityOut(key=c.key, name=c.name, lat=c.lat, lon=c.lon, stations=list(c.stations)) for c in config.CITIES.values()
    ]


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    city: str = Query(..., examples=["chiang_mai"], description="city key from /cities"),
    forecast_fn: ForecastFn = Depends(get_forecast_fn),
) -> dict[str, Any]:
    if city not in config.CITIES:
        raise _unknown_city(city)
    try:
        return forecast_fn(city)
    except live.LiveDataError as exc:
        log.warning("forecast %s failed: %s", city, exc)
        raise HTTPException(503, f"upstream data unavailable: {exc}") from exc


@app.get("/backtest", response_model=BacktestResponse)
def backtest(
    city: str = Query(..., examples=["chiang_mai"]),
    horizon: int = Query(1, ge=1, le=max(config.HORIZONS)),
    start: date | None = Query(None, description="first target date (inclusive)"),
    end: date | None = Query(None, description="last target date (inclusive)"),
    model: str | None = Query(None, description=f"model name in the backtest file (default {DEFAULT_BACKTEST_MODEL})"),
    path: Path = Depends(get_backtest_path),
) -> dict[str, Any]:
    if city not in config.CITIES:
        raise _unknown_city(city)
    try:
        preds = load_backtest(path)
        return backtest_view(preds, city, horizon, start, end, model)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

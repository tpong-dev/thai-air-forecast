"""Live forecast: today's panel rows from live APIs -> features -> saved models -> JSON-ready dict.

    python -m thaiair.live --city chiang_mai
    python -m thaiair.live --all --out live_forecast.json

The live path reuses the exact training pipeline (`features.make_features` on panel-shaped rows), so
a forecast served here is computed the same way as one scored in the backtest. Only the data
source changes: Air4Thai hourly service, Open-Meteo forecast and CAMS forecast instead of the
processed archive files.

Robustness rules (the app must always start and should degrade, not crash):
- PM2.5 is required. Without it there is nothing to persist or to lag, so a clear error is raised.
- Weather or CAMS failing is tolerated: their columns stay NaN (LightGBM handles missing values)
  and the failure is listed in `warnings`.
- Saved models missing -> a Persistence baseline fitted on the recent rows, labelled as a fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thaiair import config
from thaiair.synthetic import PANEL_COLS

log = logging.getLogger(__name__)

PAST_DAYS = 45  # enough history for the longest lag / rolling window in the features
CACHE_TTL_S = 3600  # sources update hourly at best; one fetch per city per hour is plenty
RECENT_DAYS = 14  # observed days returned for the chart
SERVED_MODEL = os.environ.get("THAIAIR_SERVED_MODEL", "lgbm")
FALLBACK_LABEL = "persistence (fallback)"

ATTRIBUTION = (
    "PM2.5 observations: Pollution Control Department (PCD), Air4Thai. "
    "Weather: Open-Meteo.com (CC BY 4.0). "
    "Generated using Copernicus Atmosphere Monitoring Service information "
    "(CAMS global forecast via Open-Meteo, CC BY 4.0)."
)


class UnknownCityError(ValueError):
    """The city key is not in `config.CITIES`."""


class LiveDataError(RuntimeError):
    """A required live source (PM2.5) is unavailable, so no honest forecast can be made."""


# --------------------------------------------------------------------------- dates


def bangkok_today() -> pd.Timestamp:
    """Today's calendar date in Asia/Bangkok as a naive midnight timestamp (the panel's date convention)."""
    return pd.Timestamp.now(tz=config.TZ).normalize().tz_localize(None)


def issue_date_for(today: pd.Timestamp) -> pd.Timestamp:
    """Live issue date = yesterday: the last day whose 24 h mean is complete."""
    return today - pd.Timedelta(days=1)


# --------------------------------------------------------------------------- source adapters
# Thin wrappers with lazy imports: the API module must import even if a data module is broken, and
# tests monkeypatch these names instead of the real network functions.


def _fetch_pm25(city: str, days: int) -> pd.DataFrame:
    from thaiair import pm25

    return pm25.recent_daily(city, days=days)


def _fetch_weather(city: str, past_days: int) -> pd.DataFrame:
    from thaiair import weather

    return weather.live_daily(city, past_days=past_days)


def _fetch_cams(city: str, past_days: int) -> pd.DataFrame:
    from thaiair import cams

    return cams.live_daily(city, past_days=past_days)


def _load_model(name: str, horizon: int) -> Any:
    from thaiair.models import registry

    return registry.load(name, horizon)


def available_models() -> list[str]:
    """Saved models found in `models/`, as `name_hN` strings. Empty if none or the registry is broken."""
    try:
        from thaiair.models import registry

        return [f"{name}_h{h}" for name, h in registry.available()]
    except Exception as exc:  # health must answer even when the registry cannot load
        log.warning("model registry unavailable: %s", exc)
        return []


# --------------------------------------------------------------------------- panel


def _check_city(city: str) -> config.City:
    if city not in config.CITIES:
        raise UnknownCityError(f"unknown city {city!r}; valid keys: {', '.join(config.CITIES)}")
    return config.CITIES[city]


def _normalise(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Coerce a source frame to (city, naive date) keys so outer joins line up."""
    df = df.copy()
    df["city"] = city
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(config.TZ).dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    return df.drop_duplicates(["city", "date"], keep="last")


def _optional_source(fetch: Callable[[], pd.DataFrame], label: str, warnings: list[str]) -> pd.DataFrame | None:
    """Run a non-essential fetch; on failure record a warning and return None (its columns become NaN)."""
    try:
        df = fetch()
    except Exception as exc:  # any upstream failure degrades to NaN features
        log.warning("%s unavailable: %s", label, exc)
        warnings.append(f"{label} unavailable ({type(exc).__name__}); its features are missing")
        return None
    if df is None or df.empty:
        warnings.append(f"{label} returned no rows; its features are missing")
        return None
    return df


def build_live_panel(
    city: str,
    today: pd.Timestamp | None = None,
    past_days: int = PAST_DAYS,
) -> pd.DataFrame:
    """Panel rows (same columns as `dataset.build_panel`) for today-past_days .. today+3.

    Every calendar day gets a row, so lag features see explicit NaN for outages instead of
    silently skipping a day. Source failures for weather / CAMS are listed in `df.attrs["warnings"]`.
    """
    _check_city(city)
    today = (today or bangkok_today()).normalize()
    warnings: list[str] = []

    try:
        pm = _fetch_pm25(city, past_days)
    except Exception as exc:  # re-raised as a clear, typed error
        raise LiveDataError(f"PM2.5 data for {city} is unavailable from Air4Thai: {exc}") from exc
    if pm is None or pm.empty or pm["pm25"].notna().sum() == 0:
        raise LiveDataError(f"Air4Thai returned no PM2.5 observations for {city} in the last {past_days} days")

    calendar = pd.DataFrame(
        {"city": city, "date": pd.date_range(today - pd.Timedelta(days=past_days), today + pd.Timedelta(days=3))}
    )
    panel = calendar.merge(_normalise(pm, city), on=["city", "date"], how="left")
    sources = {
        "weather (Open-Meteo)": lambda: _fetch_weather(city, past_days),
        "CAMS (Open-Meteo air quality)": lambda: _fetch_cams(city, past_days),
    }
    for label, fetch in sources.items():
        df = _optional_source(fetch, label, warnings)
        if df is not None:
            df = _normalise(df, city)
            new_cols = [c for c in df.columns if c in PANEL_COLS and c not in panel.columns]
            panel = panel.merge(df[["city", "date", *new_cols]], on=["city", "date"], how="left")

    for col in PANEL_COLS:
        if col not in panel.columns:
            panel[col] = np.nan
    # Days after the issue date are not observed yet, whatever a source returned for them.
    panel.loc[panel["date"] > issue_date_for(today), "pm25"] = np.nan
    panel["n_stations"] = panel["n_stations"].fillna(0).astype(int)
    panel = panel[PANEL_COLS].sort_values(["city", "date"]).reset_index(drop=True)
    panel.attrs["warnings"] = warnings
    return panel


# --------------------------------------------------------------------------- prediction


def _naive_persistence(panel: pd.DataFrame, issue: pd.Timestamp, horizon: int) -> dict[str, float]:
    """Last observed value, with a p10-p90 band from the recent h-day changes.

    Used only when the feature pipeline or the baselines module cannot be imported, so the app
    still answers with something honest and clearly labelled.
    """
    obs = panel.set_index("date")["pm25"].loc[:issue]
    last = obs.dropna()
    p50 = float(last.iloc[-1]) if len(last) else math.nan
    ratio = (obs.shift(-horizon) / obs).loc[: issue - pd.Timedelta(days=horizon)].dropna()
    if len(ratio) >= 10 and not math.isnan(p50):
        lo, hi = np.quantile(ratio, [0.1, 0.9])
        return {"p10": p50 * float(lo), "p50": p50, "p90": p50 * float(hi)}
    return {"p10": math.nan, "p50": p50, "p90": math.nan}


def _fit_persistence(train: pd.DataFrame) -> Any:
    """Persistence with intervals from its own residuals on the last ~45 days (one horizon per instance)."""
    from thaiair.baselines import Persistence

    return Persistence().fit(train)


def _predict_horizon(panel: pd.DataFrame, issue: pd.Timestamp, horizon: int) -> tuple[dict[str, float], str]:
    """(p10/p50/p90, served model label) for one horizon, falling back step by step."""
    try:
        from thaiair import features
    except ImportError as exc:
        log.warning("features module unavailable (%s); using naive persistence", exc)
        return _naive_persistence(panel, issue, horizon), FALLBACK_LABEL

    feats = features.make_features(panel, horizon)
    row = feats[pd.to_datetime(feats["issue_date"]) == issue]
    if row.empty:
        raise LiveDataError(f"no feature row for issue date {issue.date()} (horizon {horizon})")

    try:
        model, label = _load_model(SERVED_MODEL, horizon), SERVED_MODEL
    except Exception as exc:  # missing or unreadable model file -> fallback
        log.warning("model %s_h%d not loaded (%s); using persistence", SERVED_MODEL, horizon, exc)
        train = feats[feats["y"].notna() & (pd.to_datetime(feats["target_date"]) <= issue)]
        try:
            model, label = _fit_persistence(train), FALLBACK_LABEL
        except ImportError:
            return _naive_persistence(panel, issue, horizon), FALLBACK_LABEL

    pred = model.predict(row).iloc[0]
    return {q: float(pred[q]) for q in ("p10", "p50", "p90")}, label


def _clean_quantiles(q: dict[str, float]) -> dict[str, float | None]:
    """Clip at zero, keep p10 <= p50 <= p90 (independent quantile models can cross) and NaN -> None for JSON."""
    vals = {k: (max(0.0, v) if v is not None and not math.isnan(v) else None) for k, v in q.items()}
    if vals["p50"] is not None:
        if vals["p10"] is not None:
            vals["p10"] = min(vals["p10"], vals["p50"])
        if vals["p90"] is not None:
            vals["p90"] = max(vals["p90"], vals["p50"])
    return {k: (round(v, 1) if v is not None else None) for k, v in vals.items()}


def _num(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 1)


def _forecast_entries(panel: pd.DataFrame, issue: pd.Timestamp) -> tuple[list[dict[str, Any]], list[str]]:
    by_date = panel.set_index("date")
    entries, labels = [], []
    for h in config.HORIZONS:
        target = issue + pd.Timedelta(days=h)
        q, label = _predict_horizon(panel, issue, h)
        q = _clean_quantiles(q)
        p50 = q["p50"]
        entries.append(
            {
                "date": target.date().isoformat(),
                "horizon": h,
                **q,
                "aqi_band": config.thai_aqi_band(p50) if p50 is not None else None,
                "unhealthy": bool(p50 is not None and p50 > config.UNHEALTHY_PM25),
                "cams": _num(by_date["cams_pm25"].get(target)),
                "model": label,
            }
        )
        labels.append(label)
    return entries, labels


def _recent(panel: pd.DataFrame, issue: pd.Timestamp) -> list[dict[str, Any]]:
    window = panel[(panel["date"] > issue - pd.Timedelta(days=RECENT_DAYS)) & (panel["date"] <= issue)]
    return [{"date": d.date().isoformat(), "pm25": _num(v)} for d, v in zip(window["date"], window["pm25"])]


def _served_label(labels: list[str]) -> str:
    """One label if every horizon used the same model, else a per-horizon summary."""
    unique = list(dict.fromkeys(labels))
    if len(unique) == 1:
        return unique[0]
    return ", ".join(f"h{h}: {lab}" for h, lab in zip(config.HORIZONS, labels))


def _compute(city: str, today: pd.Timestamp | None) -> dict[str, Any]:
    info = _check_city(city)
    today = (today or bangkok_today()).normalize()
    issue = issue_date_for(today)
    panel = build_live_panel(city, today=today)
    entries, labels = _forecast_entries(panel, issue)
    return {
        "city": city,
        "name": info.name,
        "lat": info.lat,
        "lon": info.lon,
        "issue_date": issue.date().isoformat(),
        "generated_at": pd.Timestamp.now(tz=config.TZ).isoformat(timespec="seconds"),
        "model": _served_label(labels),
        "forecasts": entries,
        "recent": _recent(panel, issue),
        "warnings": list(panel.attrs.get("warnings", [])),
        "attribution": ATTRIBUTION,
    }


# --------------------------------------------------------------------------- public API + cache

_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def clear_cache() -> None:
    _cache.clear()


def forecast(city: str, today: pd.Timestamp | None = None, use_cache: bool = True) -> dict[str, Any]:
    """Day +1/+2/+3 forecast for one city (see module docstring). Cached in memory for an hour.

    Raises UnknownCityError for a bad key and LiveDataError when PM2.5 cannot be fetched.
    """
    _check_city(city)
    key = (city, (today or bangkok_today()).date().isoformat())
    hit = _cache.get(key)
    if use_cache and hit is not None and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1]
    result = _compute(city, today)
    _cache[key] = (time.monotonic(), result)
    return result


def forecast_all(cities: list[str] | None = None, today: pd.Timestamp | None = None) -> list[dict[str, Any]]:
    """Forecast several cities; a failing city yields `{"city", "name", "error"}` instead of stopping the rest."""
    out = []
    for city in cities or list(config.CITIES):
        try:
            out.append(forecast(city, today=today))
        except (LiveDataError, UnknownCityError) as exc:
            log.error("%s: %s", city, exc)
            name = config.CITIES[city].name if city in config.CITIES else city
            out.append({"city": city, "name": name, "error": str(exc)})
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live PM2.5 forecast for Thai cities (day +1..+3)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", action="append", choices=list(config.CITIES), help="city key (repeatable)")
    group.add_argument("--all", action="store_true", help="all cities in config.CITIES")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    results = forecast_all(None if args.all else args.city)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        log.info("wrote %d forecasts to %s", len(results), args.out)
    else:
        print(text)
    # Non-zero only if nothing could be forecast, so a scheduled run still uploads partial results.
    return 1 if all("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

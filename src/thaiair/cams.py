"""CAMS global PM2.5 forecast per city, as archived by Open-Meteo: the baseline to beat.

Open-Meteo's Air Quality API keeps, for every past hour, the value from the most recent CAMS run
(``pm2_5_previous_dayN`` exists but is empty for this domain). The archive is therefore a stitched
series of short-lead forecasts, which makes CAMS look *better* at 48/72 h than a real 72 h CAMS
forecast would be. ``docs/weather_cams.md`` explains why that is a conservative comparison.

Run ``python -m thaiair.cams`` to download (cached) and build ``data/processed/cams_daily.parquet``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from thaiair import config
from thaiair.weather import (
    CALLS,
    MIN_VALID_FRACTION,
    RAW_DIR,
    base_params,
    cached_json,
    complete_calendar,
    daily_stat,
    date_chunks,
    hourly_frame,
    print_coverage,
    today_bangkok,
)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
CAMS_CACHE = RAW_DIR / "cams"
# Requested from here; the first non-null values for Thailand appear in early August 2022.
CAMS_START = "2022-07-01"
CHUNK_MONTHS = 3
# Be explicit: without it the API may pick a regional domain (CAMS Europe) or change the default.
DOMAIN = "cams_global"


def _cams_params(city_key: str) -> dict[str, Any]:
    params = base_params(city_key)
    params.pop("wind_speed_unit")  # not a weather request
    return {**params, "domains": DOMAIN, "hourly": "pm2_5"}


def daily_cams(hourly: pd.DataFrame) -> pd.Series:
    """Daily mean PM2.5 (ug/m3) with the same min-hours rule as the weather (NaN if too few hours).

    CAMS global runs on a 3-hourly grid that Open-Meteo interpolates to hourly; sharing
    :func:`thaiair.weather.daily_stat` keeps one tested rule for both cases.
    """
    daily = daily_stat(hourly["pm25"], "mean", MIN_VALID_FRACTION).rename("cams_pm25")
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily


def fetch_cams_hourly(city_key: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Hourly CAMS PM2.5 for one city, in 3-month chunks cached under data/raw/openmeteo/cams/."""
    frames = []
    for c_start, c_end in date_chunks(start, end, CHUNK_MONTHS):
        params = {**_cams_params(city_key), "start_date": c_start.date().isoformat(), "end_date": c_end.date().isoformat()}
        cache = CAMS_CACHE / f"{city_key}_{c_start.date()}_{c_end.date()}.json"
        frames.append(hourly_frame(cached_json(AIR_QUALITY_URL, params, cache), {"pm2_5": "pm25"}))
    hourly = pd.concat(frames).sort_index()
    return hourly[~hourly.index.duplicated(keep="last")]


def build_cams_daily(cities: Iterable[str] | None = None, start: str = CAMS_START, end: str | None = None) -> pd.DataFrame:
    """``city, date, cams_pm25`` for every city from ``start`` to the last day with data."""
    end = end or today_bangkok().date().isoformat()
    frames = []
    for city in cities or config.CITIES:
        daily = daily_cams(fetch_cams_hourly(city, start, end)).reset_index()
        daily.insert(0, "city", city)
        frames.append(daily)
    return complete_calendar(pd.concat(frames, ignore_index=True), ["cams_pm25"], start)


def live_frame(payload: dict[str, Any], today: pd.Timestamp, past_days: int, city_key: str) -> pd.DataFrame:
    """``city, date, cams_pm25`` for today-past_days .. today+3 from one Air Quality API response."""
    daily = daily_cams(hourly_frame(payload, {"pm2_5": "pm25"}))
    dates = pd.date_range(today - pd.Timedelta(days=past_days), today + pd.Timedelta(days=3), freq="D", name="date")
    out = daily.reindex(dates).reset_index()
    out.insert(0, "city", city_key)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out[["city", "date", "cams_pm25"]]


def live_daily(city_key: str, past_days: int = 30, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Live CAMS: past ``past_days`` days plus the forecast for today .. today+3. Not cached."""
    params = {**_cams_params(city_key), "past_days": past_days, "forecast_days": 4}
    payload = cached_json(AIR_QUALITY_URL, params, cache=None)
    return live_frame(payload, today if today is not None else today_bangkok(), past_days, city_key)


def main() -> None:
    """Build cams_daily.parquet and print coverage per city and year plus the first date with data."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    cams = build_cams_daily()
    cams.to_parquet(config.PROCESSED_DIR / "cams_daily.parquet", index=False)
    print_coverage("cams_daily", cams, ["cams_pm25"])
    print(f"\nuncached Open-Meteo requests this run: {CALLS['count']}")


if __name__ == "__main__":
    main()

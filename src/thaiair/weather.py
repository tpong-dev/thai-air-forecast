"""Daily weather per city: observed (ERA5), forecasts as they were issued (JMA GSM), and live data.

Three products, one aggregation rule:

* ``weather_past.parquet`` - ERA5 reanalysis for the day itself (``w_*``). This is what really
  happened, so it is only safe as a *lagged* feature.
* ``weather_fc.parquet`` - JMA GSM forecasts from the Open-Meteo Previous Runs API
  (``*_previous_day{h}``). ``f{h}_*`` for date ``d`` was predicted at least ``24*h`` hours before
  each hour of ``d``, so it is known at the issue time of a ``t+h`` forecast. See
  ``docs/weather_cams.md`` for the full leakage argument.
* :func:`live_daily` - the same columns for the last ``past_days`` and the next 3 days, from the
  live Forecast API, used by the app.

All requests use ``timezone=Asia/Bangkok`` so the API returns local hours and a "day" is the local
calendar day, the same day the PM2.5 target uses. Thailand has no daylight saving time, so every
local day has exactly 24 hours.

Run ``python -m thaiair.weather`` to download (cached) and build both processed files.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thaiair import config, http

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

RAW_DIR = config.RAW_DIR / "openmeteo"
ERA5_CACHE = RAW_DIR / "era5"
JMA_CACHE = RAW_DIR / "jma_previous_runs"

BANGKOK_UTC_OFFSET = 7 * 3600
# A day needs at least 20 of its 24 hours (or the same share of a coarser grid), otherwise NaN.
# Averaging a day from a handful of hours (say, only the night) would bias it.
MIN_VALID_FRACTION = 20 / 24
# Pause after every uncached request; Open-Meteo's free tier asks for polite use.
PAUSE_S = 1.0
ERA5_CHUNK_MONTHS = 12
# Previous-runs requests are slow on the server (about 30 s for a year of 15 variables), so use
# half-year chunks to stay well inside the HTTP timeout.
JMA_CHUNK_MONTHS = 6

# Open-Meteo variable -> short name used inside this module.
API_VARS: dict[str, str] = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind",
    "wind_direction_10m": "wdir",
    "precipitation": "precip",
    "boundary_layer_height": "blh",
}
FC_API_VARS = [v for v in API_VARS if v != "boundary_layer_height"]  # JMA has no BLH
PAST_COLS = ["w_temp", "w_rh", "w_wind", "w_u", "w_v", "w_precip", "w_blh", "w_blh_min"]
FC_VARS = ["temp", "rh", "wind", "u", "v", "precip"]
FC_COLS = [f"f{h}_{v}" for h in config.HORIZONS for v in FC_VARS]

# Number of HTTP requests that actually went to the network in this process (cache hits excluded).
CALLS = {"count": 0}


# --------------------------------------------------------------------------------------------
# Small pure helpers (tested without network)
# --------------------------------------------------------------------------------------------


def today_bangkok() -> pd.Timestamp:
    """Today's local date in Bangkok as a naive midnight timestamp (the project's date convention)."""
    return pd.Timestamp.now(tz=config.TZ).tz_localize(None).normalize()


def check_timezone(payload: dict[str, Any]) -> None:
    """Fail loudly if the API did not answer in Bangkok time.

    Every daily value is built by grouping hours by their local date; if the API silently fell
    back to UTC, each "day" would be shifted by 7 hours and the error would be invisible later.
    """
    offset = payload.get("utc_offset_seconds")
    if offset != BANGKOK_UTC_OFFSET:
        raise ValueError(f"expected utc_offset_seconds={BANGKOK_UTC_OFFSET}, got {offset!r}")


def hourly_frame(payload: dict[str, Any], rename: dict[str, str]) -> pd.DataFrame:
    """Turn an Open-Meteo ``hourly`` block into a DataFrame indexed by naive local time.

    ``rename`` maps API keys to output names. A key missing from the payload becomes an all-NaN
    column, so callers always get the same columns (JMA, for example, has no BLH at all).
    """
    check_timezone(payload)
    hourly = payload.get("hourly", {})
    index = pd.DatetimeIndex(pd.to_datetime(hourly.get("time", [])), name="time")
    data = {
        out: pd.to_numeric(pd.Series(hourly.get(key, [None] * len(index)), index=index), errors="coerce")
        for key, out in rename.items()
    }
    return pd.DataFrame(data, index=index).astype(float)


def wind_to_uv(speed: np.ndarray | pd.Series, direction_deg: np.ndarray | pd.Series) -> tuple[Any, Any]:
    """Convert wind speed and meteorological direction to (u, v) components.

    The direction is where the wind blows *from* (0 = from the north, 90 = from the east), so a
    north wind has v < 0. Averaging u and v, not the angle, avoids the 359 deg / 1 deg wrap-around
    problem and keeps the "where does the air come from" signal (e.g. smoke from the north-west).
    """
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def native_step_hours(series: pd.Series) -> int:
    """Typical spacing in hours between non-missing values (1 for hourly, 3 for 3-hourly data).

    Open-Meteo normally fills every hour, but if a model is ever served on a coarser grid with
    nulls in between, the min-hours rule must count against 8 (or 4) values per day, not 24.
    """
    valid = series.dropna().index
    if len(valid) < 2:
        return 1
    seconds = valid.to_numpy().astype("datetime64[s]").astype(np.int64)  # unit-safe (ns or us index)
    gaps = np.diff(seconds) / 3600.0
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 1
    values, counts = np.unique(np.round(gaps).astype(int), return_counts=True)
    step = int(values[np.argmax(counts)])
    return step if step in (1, 2, 3, 6) else 1


def daily_stat(series: pd.Series, how: str, min_fraction: float) -> pd.Series:
    """Daily statistic of one hourly series; NaN for days with too few valid values."""
    step = native_step_hours(series)
    needed = math.ceil(min_fraction * 24 / step - 1e-9)
    grouped = series.groupby(series.index.normalize())
    value = grouped.agg(how)
    return value.where(grouped.count() >= needed)


def daily_aggregate(hourly: pd.DataFrame, min_fraction: float = MIN_VALID_FRACTION) -> pd.DataFrame:
    """Aggregate local-time hourly weather to daily values.

    Input columns (any subset): temp, rh, wind, wdir, precip, blh. Output (for the inputs present):
    temp, rh, wind, u, v (daily means), precip (daily total, mm), blh (mean), blh_min (min).

    Precipitation is Open-Meteo's "sum over the preceding hour" (mm/h). The daily total is
    ``mean(valid hours) * 24``: equal to the plain sum on a complete day, and it does not
    undercount when a few hours are missing or when the series is 3-hourly with gaps in between.
    """
    out: dict[str, pd.Series] = {}
    for col in ("temp", "rh", "wind"):
        if col in hourly:
            out[col] = daily_stat(hourly[col], "mean", min_fraction)
    if "wind" in hourly and "wdir" in hourly:
        u, v = wind_to_uv(hourly["wind"], hourly["wdir"])
        out["u"] = daily_stat(u, "mean", min_fraction)
        out["v"] = daily_stat(v, "mean", min_fraction)
    if "precip" in hourly:
        out["precip"] = daily_stat(hourly["precip"], "mean", min_fraction) * 24.0
    if "blh" in hourly:
        out["blh"] = daily_stat(hourly["blh"], "mean", min_fraction)
        out["blh_min"] = daily_stat(hourly["blh"], "min", min_fraction)
    daily = pd.DataFrame(out)
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily


def date_chunks(start: str | pd.Timestamp, end: str | pd.Timestamp, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into inclusive chunks of ``months`` calendar months, the last one clipped.

    Chunks always start on the same day-of-month as ``start`` so that a cached chunk keeps the same
    start date between runs, and only the last (still open) chunk is fetched again later.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    chunks = []
    cur = start
    while cur <= end:
        nxt = cur + pd.DateOffset(months=months)
        chunks.append((cur, min(nxt - pd.Timedelta(days=1), end)))
        cur = nxt
    return chunks


def complete_calendar(df: pd.DataFrame, columns: list[str], start: str | pd.Timestamp, trim_end: bool = True) -> pd.DataFrame:
    """One row per (city, date) from ``start`` to the city's last date with any data, sorted.

    Missing days become NaN rows (the contract says "missing = NaN, not absent"); trailing days
    with no data at all (e.g. ERA5's ~5-day delay) are dropped so the file ends where data ends.
    """
    frames = []
    for city, g in df.groupby("city", sort=True):
        g = g.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]
        has_data = g[columns].notna().any(axis=1)
        if trim_end and has_data.any():
            last = has_data[has_data].index.max()
        else:
            last = g.index.max()
        idx = pd.date_range(pd.Timestamp(start), last, freq="D", name="date")
        g = g.reindex(idx)[columns]
        g.insert(0, "city", city)
        frames.append(g.reset_index())
    if not frames:
        return pd.DataFrame(columns=["city", "date", *columns])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out[["city", "date", *columns]].sort_values(["city", "date"]).reset_index(drop=True)


def coverage_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Percent of days with a value, per city and year, for each column (for the CLI printout)."""
    tmp = df.assign(year=df["date"].dt.year)
    cov = tmp.groupby(["city", "year"])[list(columns)].agg(lambda s: 100.0 * s.notna().mean())
    return cov.round(1)


# --------------------------------------------------------------------------------------------
# Network + cache
# --------------------------------------------------------------------------------------------


def base_params(city_key: str) -> dict[str, Any]:
    city = config.CITIES[city_key]
    return {
        "latitude": city.lat,
        "longitude": city.lon,
        "timezone": config.TZ,
        "wind_speed_unit": "ms",
    }


def cached_json(url: str, params: dict[str, Any], cache: Path | None, attempts: int = 3) -> dict[str, Any]:
    """``http.get_json`` with a polite pause and a count of the requests that hit the network.

    The previous-runs server sometimes answers a heavy request with an empty, non-JSON body
    (seen once in the full download). ``http.get`` only retries on HTTP errors, so a body that is
    not JSON is retried here after a longer wait; a bad body is never written to the cache.
    """
    for attempt in range(attempts):
        if cache is None or not cache.exists():
            CALLS["count"] += 1
        try:
            return http.get_json(url, params=params, cache=cache, pause=PAUSE_S)
        except ValueError:  # requests' JSONDecodeError is a ValueError
            if attempt == attempts - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def _fetch_chunked(
    url: str,
    city_key: str,
    extra: dict[str, Any],
    rename: dict[str, str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    months: int,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download [start, end] in chunks (each cached as raw JSON) and return one hourly frame."""
    frames = []
    for c_start, c_end in date_chunks(start, end, months):
        params = {
            **base_params(city_key),
            **extra,
            "start_date": c_start.date().isoformat(),
            "end_date": c_end.date().isoformat(),
        }
        cache = cache_dir / f"{city_key}_{c_start.date()}_{c_end.date()}.json"
        frames.append(hourly_frame(cached_json(url, params, cache), rename))
    hourly = pd.concat(frames).sort_index()
    return hourly[~hourly.index.duplicated(keep="last")]


def fetch_era5_hourly(city_key: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Hourly ERA5 for one city. ``models=era5`` is explicit so the source never silently changes.

    (The archive's default "best_match" blends ERA5-Land and, for the last days, an operational
    model, which would mix two products in one feature.)
    """
    extra = {"models": "era5", "hourly": ",".join(API_VARS)}
    return _fetch_chunked(ERA5_URL, city_key, extra, API_VARS, start, end, ERA5_CHUNK_MONTHS, ERA5_CACHE)


def jma_rename(horizon: int) -> dict[str, str]:
    """API keys of the ``previous_day{horizon}`` JMA variables -> short names."""
    return {f"{v}_previous_day{horizon}": API_VARS[v] for v in FC_API_VARS}


def fetch_jma_hourly(city_key: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Hourly JMA GSM previous-run forecasts for h = 1..3, columns ``{short}_h{h}`` (e.g. temp_h2)."""
    rename = {key: f"{short}_h{h}" for h in config.HORIZONS for key, short in jma_rename(h).items()}
    extra = {"models": "jma_gsm", "hourly": ",".join(rename)}
    return _fetch_chunked(PREVIOUS_RUNS_URL, city_key, extra, rename, start, end, JMA_CHUNK_MONTHS, JMA_CACHE)


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def past_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly ERA5 (short names) -> daily ``w_*`` columns in contract order."""
    daily = daily_aggregate(hourly).add_prefix("w_")
    return daily.reindex(columns=PAST_COLS)


def forecast_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly JMA frame with ``{short}_h{h}`` columns -> daily ``f{h}_*`` columns in contract order."""
    parts = []
    for h in config.HORIZONS:
        cols = {c: c.rsplit("_h", 1)[0] for c in hourly.columns if c.endswith(f"_h{h}")}
        daily = daily_aggregate(hourly[list(cols)].rename(columns=cols))
        parts.append(daily.reindex(columns=FC_VARS).add_prefix(f"f{h}_"))
    return pd.concat(parts, axis=1).reindex(columns=FC_COLS)


def _with_city(daily: pd.DataFrame, city_key: str) -> pd.DataFrame:
    out = daily.reset_index()
    out.insert(0, "city", city_key)
    return out


def build_weather_past(cities: Iterable[str] | None = None, start: str = config.DATA_START, end: str | None = None) -> pd.DataFrame:
    """Daily ERA5 weather for every city (``city, date, w_*``)."""
    end = end or (today_bangkok() - pd.Timedelta(days=1)).date().isoformat()
    frames = [_with_city(past_daily(fetch_era5_hourly(c, start, end)), c) for c in cities or config.CITIES]
    return complete_calendar(pd.concat(frames, ignore_index=True), PAST_COLS, start)


def build_weather_fc(cities: Iterable[str] | None = None, start: str = config.DATA_START, end: str | None = None) -> pd.DataFrame:
    """Daily JMA GSM issued forecasts for every city (``city, date, f1_*..f3_*``)."""
    end = end or today_bangkok().date().isoformat()
    frames = [_with_city(forecast_daily(fetch_jma_hourly(c, start, end)), c) for c in cities or config.CITIES]
    return complete_calendar(pd.concat(frames, ignore_index=True), FC_COLS, start)


# --------------------------------------------------------------------------------------------
# Live (used by thaiair.live)
# --------------------------------------------------------------------------------------------

LIVE_MODELS = "jma_gsm,best_match"


def live_frame(payload: dict[str, Any], today: pd.Timestamp, past_days: int, city_key: str) -> pd.DataFrame:
    """Build the live ``city, date, w_*, f1_*..f3_*`` frame from one Forecast API response.

    * ``w_*`` (dates before today): JMA GSM past hours from the Forecast API, plus BLH from
      ``best_match`` because JMA has no BLH. This is not ERA5, so it is a slightly different
      product than in training (documented in docs/weather_cams.md).
    * ``f1_*``, ``f2_*``, ``f3_*`` (today .. today+3): the latest JMA run, copied to all three
      prefixes. That is fresher than ``previous_day{h}`` in training, which can only help.
    """
    jma = hourly_frame(payload, {f"{k}_jma_gsm": v for k, v in API_VARS.items() if k != "boundary_layer_height"})
    blh = hourly_frame(payload, {"boundary_layer_height_best_match": "blh"})
    daily = daily_aggregate(jma.join(blh))
    dates = pd.date_range(today - pd.Timedelta(days=past_days), today + pd.Timedelta(days=3), freq="D", name="date")
    daily = daily.reindex(dates)
    past = dates < today
    out = pd.DataFrame(index=dates)
    for col in PAST_COLS:
        out[col] = daily[col.removeprefix("w_")].where(past)
    for h in config.HORIZONS:
        for v in FC_VARS:
            out[f"f{h}_{v}"] = daily[v].where(~past)
    out = _with_city(out, city_key)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out[["city", "date", *PAST_COLS, *FC_COLS]]


def live_daily(city_key: str, past_days: int = 30, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Live weather for ``today - past_days`` .. ``today + 3`` (contract section 8). Not cached."""
    params = {
        **base_params(city_key),
        "models": LIVE_MODELS,
        "hourly": ",".join(API_VARS),
        "past_days": past_days,
        "forecast_days": 4,
    }
    payload = cached_json(FORECAST_URL, params, cache=None)
    return live_frame(payload, today if today is not None else today_bangkok(), past_days, city_key)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def first_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """First date with a value, per city (rows) and column."""
    rows = {city: {c: g.loc[g[c].notna(), "date"].min() for c in columns} for city, g in df.groupby("city")}
    return pd.DataFrame(rows).T


def print_coverage(name: str, df: pd.DataFrame, columns: list[str]) -> None:
    """Print row count, date range, % of days with data per city and year, and first dates."""
    print(f"\n{name}: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}")
    cov = coverage_table(df, columns)
    for col in columns:
        print(f"{col}: % of days with a value\n" + cov[col].unstack("city").to_string())
    print("first date with a value\n" + first_dates(df, columns).to_string())


def main(argv: list[str] | None = None) -> None:
    """Build weather_past.parquet and weather_fc.parquet and print their coverage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["past", "fc"], help="build just one of the two files")
    args = parser.parse_args(argv)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    if args.only in (None, "past"):
        past = build_weather_past()
        past.to_parquet(config.PROCESSED_DIR / "weather_past.parquet", index=False)
        print_coverage("weather_past", past, ["w_temp", "w_blh"])
    if args.only in (None, "fc"):
        fc = build_weather_fc()
        fc.to_parquet(config.PROCESSED_DIR / "weather_fc.parquet", index=False)
        print_coverage("weather_fc", fc, ["f1_temp", "f2_temp", "f3_temp"])
    print(f"\nuncached Open-Meteo requests this run: {CALLS['count']}")


if __name__ == "__main__":
    main()

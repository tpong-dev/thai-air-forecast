"""PM2.5 ground truth: download, parse, clean and aggregate Air4Thai station data to city daily means.

Two sources are stitched together:

* the PCD yearly "daily" files (``{year}.xlsx``, one row per day, one column per station, 24 h means),
  which cover 2019-01-01 to 2026-06-30, and
* the Air4Thai hourly history service, which only keeps the latest ~2 months. Its hourly values are
  turned into daily means here, so the series can continue past the last yearly file.

All dates are Thai local calendar days (Asia/Bangkok, UTC+7, no daylight saving). They are stored as
naive midnight timestamps, because every other file in the project uses the same convention.

The target is never imputed: a day without enough valid data stays NaN. See ``docs/cleaning.md``.

Run ``python -m thaiair.pm25`` to build ``data/processed/pm25_daily.parquet`` and
``data/processed/pm25_station_daily.parquet``.
"""

from __future__ import annotations

import json
import logging
import ssl
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thaiair import config, http

log = logging.getLogger(__name__)

PCD_DAILY_URL = "https://air4thai.net/daily_download/PM2.5/{year}.xlsx"
HOURLY_URL = "https://air4thai.com/forweb/getHistoryData.php"
LATEST_URL = "https://air4thai.pcd.go.th/services/getNewAQI_JSON.php"

FIRST_YEAR = 2019
PAUSE = 0.5  # seconds between requests: the Air4Thai servers are a public service, be polite

# A daily mean needs at least 18 of 24 valid hours (75 %). This is the completeness rule used by
# the US EPA for 24 h PM2.5 and by most national networks; with fewer hours one missing night or
# rush hour can bias the mean by a lot.
MIN_VALID_HOURS = 18

# Physical range for a PM2.5 concentration (ug/m3). Below 0 is impossible; above 1000 is beyond the
# range of the beta-attenuation monitors used by PCD, so such numbers are error codes, not air.
MIN_PM25 = 0.0
MAX_PM25 = 1000.0

# Stuck sensor: the same daily value on this many consecutive days. Daily means of a working monitor
# move every day; the longest honest run seen in 2019-2026 is 6 days, on integer-rounded values in
# the rainy season. 7 or more identical days is treated as a frozen logger.
STUCK_MIN_RUN = 7

# Isolated spike: a day more than SPIKE_RATIO times the larger neighbour AND more than SPIKE_MIN_JUMP
# above it, with both neighbours present. Burning-season peaks build up over days and fireworks or
# local fires rarely triple a 24 h mean, so this only catches one-day glitches. It is strict on purpose.
SPIKE_RATIO = 4.0
SPIKE_MIN_JUMP = 100.0

SOURCES = ("pcd_daily", "a4t_hourly")
DAILY_COLS = ["city", "date", "pm25", "n_stations"]
STATION_COLS = ["station", "date", "pm25", "source"]


# --------------------------------------------------------------------------------------------------
# HTTP setup
# --------------------------------------------------------------------------------------------------

_trust_store_ready = False


def _use_os_trust_store() -> None:
    """Verify Air4Thai TLS certificates with the operating system's trust store.

    WHY: the Air4Thai servers send an incomplete certificate chain (a Let's Encrypt leaf with the
    wrong intermediates). Browsers and the Windows/macOS verifier fetch the missing intermediate, but
    the ``certifi`` bundle used by ``requests`` cannot, so plain ``requests`` fails with
    CERTIFICATE_VERIFY_FAILED. ``truststore`` delegates verification to the OS, which keeps TLS
    verification on. The adapter is mounted only for ``https://air4thai.`` URLs, so other sources
    are untouched. If ``truststore`` is unavailable nothing changes.
    """
    global _trust_store_ready
    if _trust_store_ready:
        return
    try:
        import truststore  # type: ignore[import-not-found]
    except ImportError:
        try:
            from pip._vendor import truststore  # type: ignore[no-redef]
        except ImportError:
            log.warning("truststore not available; Air4Thai HTTPS may fail certificate checks")
            _trust_store_ready = True
            return
    from requests.adapters import HTTPAdapter

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    class _OsTrustAdapter(HTTPAdapter):
        def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
            kwargs["ssl_context"] = context
            super().init_poolmanager(*args, **kwargs)

    http.session().mount("https://air4thai.", _OsTrustAdapter())
    _trust_store_ready = True


def bangkok_today() -> date:
    """Today's calendar date in Thailand (UTC+7), whatever the machine's time zone is."""
    return pd.Timestamp.now(tz=config.TZ).date()


def station_ids(cities: Iterable[str] | None = None) -> list[str]:
    """Upper-case station IDs of the configured cities, in config order, without repeats."""
    keys = list(cities) if cities is not None else list(config.CITIES)
    seen: dict[str, None] = {}
    for key in keys:
        for sid in config.CITIES[key].stations:
            seen[sid.upper()] = None
    return list(seen)


def _naive_days(values: Any) -> Any:
    """Normalise to naive local midnight with nanosecond resolution (the contract dtype).

    WHY the explicit cast: pandas 3 defaults to microsecond resolution, the contract says [ns].
    Accepts a Series (returns a Series) or anything else date-like (returns a DatetimeIndex).
    """
    ts = pd.to_datetime(values)
    ts = ts.dt.normalize() if isinstance(ts, pd.Series) else pd.DatetimeIndex(ts).normalize()
    return ts.astype("datetime64[ns]")


# --------------------------------------------------------------------------------------------------
# Source 1: PCD yearly daily files
# --------------------------------------------------------------------------------------------------


@dataclass
class PcdFile:
    """Parsed yearly file: long data for every station plus notes about the quirks that were fixed."""

    data: pd.DataFrame  # station, date, pm25
    n_markers: int = 0  # text cells such as "N/A" (= no data in the source)
    notes: list[str] = field(default_factory=list)


def download_pcd_years(years: Iterable[int], raw_dir: Path | None = None) -> list[Path]:
    """Download the yearly files once into ``data/raw/pcd_daily`` (existing files are kept)."""
    raw_dir = raw_dir or config.RAW_DIR / "pcd_daily"
    paths = []
    for year in years:
        dest = raw_dir / f"{year}.xlsx"
        if not (dest.exists() and dest.stat().st_size > 0):
            _use_os_trust_store()
            log.info("downloading PCD daily file %s", year)
        try:
            paths.append(http.download(PCD_DAILY_URL.format(year=year), dest, pause=PAUSE))
        except Exception as exc:  # a missing future year must not stop the build
            log.warning("could not download %s: %s", year, exc)
    return paths


def _find_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    """Index of the row whose first cell is "Date". WHY: guards against title rows above the header."""
    for i in range(min(max_scan, len(raw))):
        first = raw.iat[i, 0]
        if isinstance(first, str) and first.strip().lower() == "date":
            return i
    raise ValueError("no 'Date' header row found in the first rows of the sheet")


def _to_number(value: Any) -> float:
    """Cell value as float; text markers ("N/A", "n/a", "-") and blanks become NaN."""
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return np.nan
    return np.nan


def _is_marker(value: Any) -> bool:
    """True for non-empty text that is not a number (the files use "N/A" / "n/a" for missing)."""
    return isinstance(value, str) and value.strip() != "" and np.isnan(_to_number(value))


def fix_duplicate_dates(dates: pd.Series) -> tuple[pd.Series, list[str]]:
    """Repair a mislabelled date, or drop dates that cannot be repaired.

    WHY: the 2021 file lists 2021-06-30 twice and has no 2021-06-29. The rows are in order, so the
    first of the two is the 29th with a typo. When the rows form a gap-free daily sequence by
    position and only the duplicated rows disagree with it, the date is rebuilt from the position.
    Otherwise every row of a duplicated date is dropped (we cannot tell which one is right).
    """
    if not dates.duplicated().any():
        return dates, []
    positional = pd.Series(pd.date_range(dates.iloc[0], periods=len(dates), freq="D"), index=dates.index)
    differs = positional != dates
    dup = dates.duplicated(keep=False)
    if dates.is_monotonic_increasing and not (differs & ~dup).any():
        notes = [f"relabelled {d.date()} -> {p.date()} (duplicate date, rebuilt from row order)"
                 for d, p in zip(dates[differs], positional[differs])]
        return positional, notes
    bad = sorted({d.date() for d in dates[dup]})
    return dates[~dup], [f"dropped duplicate dates that could not be repaired: {bad}"]


def parse_pcd_sheet(raw: pd.DataFrame, year: int | None = None) -> PcdFile:
    """Turn the raw first sheet (header=None, object cells) into long ``station, date, pm25`` rows.

    Handles: header not on the first row, station IDs with odd case or spaces, duplicated station
    columns, Thai footnote rows and blank rows at the bottom, text markers, duplicated dates and rows
    from another year.
    """
    h = _find_header_row(raw)
    header = [str(c).strip().upper() if c is not None and not pd.isna(c) else "" for c in raw.iloc[h]]
    body = raw.iloc[h + 1 :].reset_index(drop=True)
    dates = pd.to_datetime(body.iloc[:, 0].map(_cell_to_date), errors="coerce")
    body = body[dates.notna()]
    dates = dates[dates.notna()].dt.normalize()
    notes: list[str] = []
    if year is not None:
        other = dates.dt.year != year
        if other.any():
            notes.append(f"dropped {int(other.sum())} rows dated outside {year}")
            body, dates = body[~other], dates[~other]
    dates, fix_notes = fix_duplicate_dates(dates.reset_index(drop=True))
    notes += fix_notes
    body = body.reset_index(drop=True).loc[dates.index]

    frames, n_markers = [], 0
    for j, sid in enumerate(header):
        if j == 0 or not sid:
            continue
        cells = body.iloc[:, j]
        n_markers += int(cells.map(_is_marker).sum())
        frames.append(pd.DataFrame({"station": sid, "date": dates.to_numpy(), "pm25": cells.map(_to_number).to_numpy()}))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["station", "date", "pm25"])
    data["pm25"] = data["pm25"].astype(float)
    if data.duplicated(["station", "date"]).any():
        notes.append("merged duplicated station columns (first non-missing value kept)")
        data = data.sort_values("pm25", na_position="last", kind="stable").drop_duplicates(["station", "date"])
    data["date"] = _naive_days(data["date"])
    return PcdFile(data.sort_values(["station", "date"]).reset_index(drop=True), n_markers, notes)


def _cell_to_date(value: Any) -> Any:
    """Keep real dates and date-like strings; everything else (footnotes, blanks) becomes NaT."""
    if isinstance(value, datetime | date | pd.Timestamp):
        return value
    if isinstance(value, str):
        return pd.to_datetime(value.strip(), errors="coerce", dayfirst=False)
    return pd.NaT


def read_pcd_xlsx(path: Path) -> PcdFile:
    """Read one yearly file. The sheet name changes between years ("PM2.5", "Data", "DATA"), so the
    first sheet is used; the second sheet is a station description table."""
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object, na_filter=False)
    raw = raw.replace("", None)
    year = int(path.stem) if path.stem.isdigit() else None
    return parse_pcd_sheet(raw, year)


# --------------------------------------------------------------------------------------------------
# Source 2: Air4Thai hourly history service
# --------------------------------------------------------------------------------------------------


def parse_hourly_json(doc: Any) -> pd.DataFrame:
    """``{"stations": [{"stationID", "data": [{"DATETIMEDATA", "PM25"}]}]}`` -> ``station, time, pm25``."""
    rows = []
    for st in (doc or {}).get("stations") or []:
        sid = str(st.get("stationID", "")).strip().upper()
        for rec in st.get("data") or []:
            rows.append((sid, rec.get("DATETIMEDATA"), _to_number(rec.get("PM25"))))
    df = pd.DataFrame(rows, columns=["station", "time", "pm25"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["pm25"] = df["pm25"].astype(float)
    return df[df["time"].notna()].reset_index(drop=True)


def hourly_to_daily(hourly: pd.DataFrame, min_hours: int = MIN_VALID_HOURS) -> pd.DataFrame:
    """Daily mean per station from hourly values, only for days with ``>= min_hours`` valid hours.

    Hours outside [MIN_PM25, MAX_PM25] (error codes such as -1 or 9999) count as missing. The day is
    the local calendar day of the timestamp (hours 00..23). Means are rounded to 0.1 like the PCD files.
    Returns ``station, date, pm25, n_hours``.
    """
    df = hourly.dropna(subset=["time"]).drop_duplicates(["station", "time"], keep="last").copy()
    df.loc[(df["pm25"] < MIN_PM25) | (df["pm25"] > MAX_PM25), "pm25"] = np.nan
    df["date"] = _naive_days(df["time"])
    out = df.groupby(["station", "date"], as_index=False).agg(pm25=("pm25", "mean"), n_hours=("pm25", "count"))
    out = out[out["n_hours"] >= min_hours].copy()
    out["pm25"] = out["pm25"].round(1)
    out["n_hours"] = out["n_hours"].astype(int)
    return out.reset_index(drop=True)


def _month_starts(start: date, end: date) -> list[date]:
    first = date(start.year, start.month, 1)
    out = []
    while first <= end:
        out.append(first)
        first = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    return out


def _request_hourly(station: str, sdate: date, edate: date) -> Any:
    params = {"stationID": station.lower(), "param": "PM25", "type": "hr",
              "sdate": sdate.isoformat(), "edate": edate.isoformat(), "stime": "00", "etime": "23"}
    _use_os_trust_store()
    return http.get_json(HOURLY_URL, params=params, pause=PAUSE)


def fetch_hourly(station: str, start: date, end: date, raw_dir: Path | None = None) -> pd.DataFrame:
    """Hourly PM2.5 of one station for ``start..end`` (inclusive), cached per calendar month.

    WHY per month: the service forgets data after ~2 months, so a cached month is the only copy we
    will ever have. A month is cached for good once it ended more than 2 days ago (late uploads);
    younger months are fetched again on every run and the cache file is overwritten.
    With ``raw_dir=None`` nothing is cached (used by the live app).
    """
    frames = []
    stale_before = bangkok_today() - timedelta(days=2)
    for m0 in _month_starts(start, end):
        m1 = (pd.Timestamp(m0) + pd.offsets.MonthEnd(0)).date()
        sdate, edate = max(m0, start), min(m1, end)
        cache = raw_dir / f"{station.upper()}_{m0:%Y-%m}.json" if raw_dir is not None else None
        complete = m1 < stale_before and sdate == m0 and edate == m1
        if cache is not None and complete and cache.exists():
            doc = json.loads(cache.read_text(encoding="utf-8"))
        else:
            doc = _request_hourly(station, sdate, edate)
            if not isinstance(doc, dict) or doc.get("result", "OK") != "OK":
                log.warning("hourly service error for %s %s: %s", station, m0, doc)
                continue
            if cache is not None and sdate == m0:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(doc), encoding="utf-8")
        frames.append(parse_hourly_json(doc))
    if not frames:
        return pd.DataFrame({"station": pd.Series(dtype=str), "time": pd.Series(dtype="datetime64[ns]"),
                             "pm25": pd.Series(dtype=float)})
    df = pd.concat(frames, ignore_index=True)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["time"] >= lo) & (df["time"] < hi)].reset_index(drop=True)


def hourly_station_daily(stations: Iterable[str], start: date, end: date, raw_dir: Path | None = None) -> pd.DataFrame:
    """Daily means (``station, date, pm25, n_hours``) from the hourly service for several stations."""
    frames = [hourly_to_daily(fetch_hourly(s, start, end, raw_dir)) for s in stations]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({"station": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                             "pm25": pd.Series(dtype=float), "n_hours": pd.Series(dtype=int)})
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------------------


def _stuck_mask(s: pd.Series, min_run: int) -> pd.Series:
    """True for every day inside a run of >= min_run identical values on consecutive days."""
    same_as_prev = s.eq(s.shift(1)) & s.notna()
    run_id = (~same_as_prev).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (run_len >= min_run) & s.notna()


def _spike_mask(s: pd.Series, ratio: float, min_jump: float) -> pd.Series:
    """True for an isolated one-day spike (both neighbours present and far lower)."""
    nb = pd.concat([s.shift(1), s.shift(-1)], axis=1).max(axis=1, skipna=False)
    return (s > ratio * nb) & (s - nb > min_jump)


def clean_station_daily(
    df: pd.DataFrame,
    stuck_min_run: int = STUCK_MIN_RUN,
    spike_ratio: float = SPIKE_RATIO,
    spike_min_jump: float = SPIKE_MIN_JUMP,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set implausible station-day values to NaN and count how many each rule removed.

    Rules, in order: out of physical range, stuck sensor, isolated spike. Values are never filled
    in: a removed value becomes NaN and the city median simply uses the other stations.
    Input and output: ``station, date, pm25`` (+ any extra columns, kept as they are).
    """
    out = df.sort_values(["station", "date"]).reset_index(drop=True).copy()
    counts = {"out_of_range": 0, "stuck": 0, "spike": 0}
    bad = (out["pm25"] < MIN_PM25) | (out["pm25"] > MAX_PM25)
    counts["out_of_range"] = int(bad.sum())
    out.loc[bad, "pm25"] = np.nan
    if out.duplicated(["station", "date"]).any():
        raise ValueError("clean_station_daily needs one row per (station, date)")
    for sid, idx in out.groupby("station").groups.items():
        part = out.loc[idx, ["date", "pm25"]]
        # On a full daily calendar, so a missing day breaks a run and is not a neighbour.
        s = part.set_index("date")["pm25"].asfreq("D")
        stuck = _stuck_mask(s, stuck_min_run)
        s2 = s.mask(stuck)
        spike = _spike_mask(s2, spike_ratio, spike_min_jump)
        drop_dates = s.index[stuck | spike]
        counts["stuck"] += int(stuck.sum())
        counts["spike"] += int(spike.sum())
        if len(drop_dates):
            log.info("%s: removed %d stuck, %d spike values", sid, int(stuck.sum()), int(spike.sum()))
            out.loc[idx[part["date"].isin(drop_dates).to_numpy()], "pm25"] = np.nan
    return out, counts


# --------------------------------------------------------------------------------------------------
# Combining and aggregating
# --------------------------------------------------------------------------------------------------


def combine_sources(pcd: pd.DataFrame, hourly_daily: pd.DataFrame) -> pd.DataFrame:
    """Station-day table from both sources. Where both cover a station-day, the PCD file wins.

    WHY: the yearly file is PCD's checked archive; the hourly service is preliminary live data.
    A PCD row counts as "covering" the day even when its value is NaN (PCD said: no valid data).
    """
    a = pcd.loc[:, ["station", "date", "pm25"]].assign(source="pcd_daily")
    b = hourly_daily.loc[:, ["station", "date", "pm25"]].assign(source="a4t_hourly")
    both = pd.concat([a, b], ignore_index=True)
    both["date"] = _naive_days(both["date"])
    both["rank"] = both["source"].map({s: i for i, s in enumerate(SOURCES)})
    both = both.sort_values(["station", "date", "rank"]).drop_duplicates(["station", "date"], keep="first")
    return both.drop(columns="rank").reset_index(drop=True)[STATION_COLS]


def city_daily(
    station_daily: pd.DataFrame,
    start: str | date | None = None,
    end: str | date | None = None,
    cities: Iterable[str] | None = None,
) -> pd.DataFrame:
    """City daily PM2.5 = median of the city's stations that reported; ``n_stations`` = how many.

    WHY median: with 9 Bangkok stations one roadside or faulty monitor cannot drag the city value;
    for 1-2 station cities it equals the station (mean of two). The output has a complete daily
    calendar per city from ``start`` (default ``config.DATA_START``, read at call time) to ``end``
    (default: last date with any value), NaN where no station reported, exactly the columns and
    dtypes of the contract.
    """
    start = start if start is not None else config.DATA_START
    keys = list(cities) if cities is not None else list(config.CITIES)
    valid = station_daily.dropna(subset=["pm25"])
    if end is None:
        end = valid["date"].max() if len(valid) else pd.Timestamp(start)
    calendar = _naive_days(pd.date_range(start, end, freq="D"))
    frames = []
    for key in keys:
        ids = [s.upper() for s in config.CITIES[key].stations]
        sub = valid[valid["station"].str.upper().isin(ids)]
        agg = sub.groupby("date")["pm25"].agg(["median", "count"]).reindex(calendar)
        frames.append(pd.DataFrame({
            "city": key,
            "date": calendar,
            "pm25": agg["median"].to_numpy(dtype=float),
            "n_stations": agg["count"].fillna(0).to_numpy(dtype="int64"),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["city"] = out["city"].astype(str)
    return out.sort_values(["city", "date"]).reset_index(drop=True)[DAILY_COLS]


def coverage_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Share of calendar days with a city value, city x year (0..1)."""
    d = daily.assign(year=daily["date"].dt.year, has=daily["pm25"].notna())
    return d.pivot_table(index="city", columns="year", values="has", aggfunc="mean").reindex(daily["city"].unique())


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------


@dataclass
class BuildResult:
    daily: pd.DataFrame
    station_daily: pd.DataFrame
    counts: dict[str, int]
    notes: list[str]


def build(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    today: date | None = None,
    years: Iterable[int] | None = None,
) -> BuildResult:
    """Download (if needed), clean and aggregate everything, and write the two parquet files."""
    raw_dir = raw_dir or config.RAW_DIR
    processed_dir = processed_dir or config.PROCESSED_DIR
    today = today or bangkok_today()
    years = list(years) if years is not None else list(range(FIRST_YEAR, today.year + 1))
    ids = station_ids()
    notes: list[str] = []
    counts: dict[str, int] = {"pcd_missing": 0}

    pcd_frames = []
    for path in download_pcd_years(years, raw_dir / "pcd_daily"):
        parsed = read_pcd_xlsx(path)
        # Station-days that the PCD file itself marks as missing ("N/A" or blank), our stations only.
        counts["pcd_missing"] += int((parsed.data["station"].isin(ids) & parsed.data["pm25"].isna()).sum())
        notes += [f"{path.name}: {n}" for n in parsed.notes]
        pcd_frames.append(parsed.data[parsed.data["station"].isin(ids)])
    pcd = pd.concat(pcd_frames, ignore_index=True) if pcd_frames else pd.DataFrame(columns=["station", "date", "pm25"])

    yesterday = today - timedelta(days=1)
    pcd_end = pcd["date"].max().date() if len(pcd) else pd.Timestamp(config.DATA_START).date() - timedelta(days=1)
    hourly = hourly_station_daily(ids, pcd_end + timedelta(days=1), yesterday, raw_dir / "a4t_hourly")
    counts["hourly_days"] = len(hourly)

    station_daily = combine_sources(pcd, hourly)
    station_daily = station_daily[station_daily["date"] >= pd.Timestamp(config.DATA_START)]
    station_daily, rule_counts = clean_station_daily(station_daily)
    counts.update(rule_counts)
    station_daily = station_daily[STATION_COLS].reset_index(drop=True)
    station_daily["date"] = _naive_days(station_daily["date"])
    daily = city_daily(station_daily)

    processed_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(processed_dir / "pm25_daily.parquet", index=False)
    station_daily.to_parquet(processed_dir / "pm25_station_daily.parquet", index=False)
    return BuildResult(daily, station_daily, counts, notes)


# --------------------------------------------------------------------------------------------------
# Live functions (used by thaiair.live)
# --------------------------------------------------------------------------------------------------


def recent_daily(city_key: str, days: int = 45, today: date | None = None) -> pd.DataFrame:
    """City daily PM2.5 for the last ``days`` complete days (today in Bangkok is excluded).

    Uses the hourly service with the same 18-hour rule and cleaning as the archive, no disk cache.
    Returns ``city, date, pm25, n_stations`` on a full calendar (NaN / 0 where nothing reported).
    """
    today = today or bangkok_today()
    start, end = today - timedelta(days=days), today - timedelta(days=1)
    ids = [s.upper() for s in config.CITIES[city_key].stations]
    hourly = hourly_station_daily(ids, start, end, raw_dir=None)
    if len(hourly):
        hourly, _ = clean_station_daily(hourly)
    return city_daily(hourly, start=start, end=end, cities=[city_key])


def parse_latest(doc: Any) -> pd.DataFrame:
    """Latest reading per station from ``getNewAQI_JSON``. The value "-1" means missing."""
    rows = []
    for st in (doc or {}).get("stations") or []:
        last = st.get("AQILast") or {}
        value = _to_number((last.get("PM25") or {}).get("value"))
        if value < 0:
            value = np.nan
        when = pd.to_datetime(f"{last.get('date', '')} {last.get('time', '')}".strip(), errors="coerce")
        rows.append({
            "station": str(st.get("stationID", "")).strip().upper(),
            "name": st.get("nameEN") or st.get("nameTH"),
            "area": st.get("areaEN") or st.get("areaTH"),
            "lat": _to_number(st.get("lat")),
            "lon": _to_number(st.get("long")),
            "pm25": value,
            "time": when,
        })
    cols = ["station", "name", "area", "lat", "lon", "pm25", "time"]
    df = pd.DataFrame(rows, columns=cols)
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ns]")
    return df


def latest_stations() -> pd.DataFrame:
    """``station, name, area, lat, lon, pm25, time`` for every Air4Thai station (for the map)."""
    _use_os_trust_store()
    return parse_latest(http.get_json(LATEST_URL))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def format_coverage(table: pd.DataFrame) -> str:
    """Coverage table as a Markdown table with percentages (also pasted into docs/cleaning.md)."""
    years = [str(c) for c in table.columns]
    lines = ["| city | " + " | ".join(years) + " |", "|---|" + "---:|" * len(years)]
    for city, row in table.iterrows():
        cells = ["" if pd.isna(v) else f"{100 * v:.0f}%" for v in row]
        lines.append(f"| {city} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = build()
    d = res.daily
    print(f"pm25_daily: {len(d)} rows, {d['date'].min().date()} .. {d['date'].max().date()}")
    print(f"pm25_station_daily: {len(res.station_daily)} rows; by source: "
          f"{res.station_daily.groupby('source')['pm25'].count().to_dict()}")
    print("rule counts (station-days):", res.counts)
    for n in res.notes:
        print("note:", n)
    print("\nCoverage (share of days with a city value):")
    print(format_coverage(coverage_table(d)))


if __name__ == "__main__":
    main()

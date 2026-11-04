"""Join the processed source files into one daily panel: one row per (city, date).

The panel is the single input of the feature builder, the backtest and the live app, so it has a
fixed shape: every city gets a complete daily calendar (missing days are NaN rows, never absent
rows) and always the same columns (`synthetic.PANEL_COLS`). A missing optional source therefore
degrades the models gracefully instead of breaking the feature code.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from thaiair import config, synthetic

log = logging.getLogger(__name__)

PM25_FILE = "pm25_daily.parquet"
# Optional sources: file name -> the panel columns it provides.
OPTIONAL_FILES: dict[str, list[str]] = {
    "weather_past.parquet": synthetic.PAST_WEATHER_COLS,
    "weather_fc.parquet": synthetic.FC_WEATHER_COLS,
    "cams_daily.parquet": synthetic.CAMS_COLS,
}
KEYS = ["city", "date"]


def _read_source(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one processed file, keep the contract columns and normalise the keys.

    Dates are forced to naive midnight timestamps so that files written by different modules
    (some may carry a time zone or a time of day) join exactly on the calendar day.
    """
    df = pd.read_parquet(path)
    missing = [c for c in [*KEYS, *columns] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing} (see docs/architecture.md section 1)")
    df = df[[*KEYS, *columns]].copy()
    df["city"] = df["city"].astype(str)
    dates = pd.to_datetime(df["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    df["date"] = dates.dt.normalize().astype("datetime64[ns]")
    dupes = df.duplicated(KEYS)
    if dupes.any():
        raise ValueError(f"{path.name} has {int(dupes.sum())} duplicate (city, date) rows")
    return df


def _last_date_with_data(frames: list[pd.DataFrame]) -> pd.Timestamp:
    """Latest date on which any source has at least one non-NaN value."""
    last = []
    for df in frames:
        values = df.drop(columns=KEYS)
        has_data = values.notna().any(axis=1)
        if has_data.any():
            last.append(df.loc[has_data, "date"].max())
    if not last:
        raise ValueError("no source file contains any data")
    return max(last)


def _calendar(cities: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Complete (city, date) grid so that a missing day is a NaN row, not an absent one."""
    dates = pd.date_range(start, end, freq="D")
    return pd.MultiIndex.from_product([cities, dates], names=KEYS).to_frame(index=False)


def _ordered_cities(frames: list[pd.DataFrame]) -> list[str]:
    """Cities present in the data, in `config.CITIES` order; unknown keys are dropped with a warning."""
    seen = set().union(*(set(df["city"].unique()) for df in frames))
    unknown = sorted(seen - set(config.CITIES))
    if unknown:
        warnings.warn(f"ignoring cities not in config.CITIES: {unknown}", stacklevel=3)
    return [c for c in config.CITIES if c in seen]


def build_panel(processed_dir: str | Path | None = None) -> pd.DataFrame:
    """Outer-join the processed files over a complete daily calendar per city.

    `pm25_daily.parquet` is required (it is the target). The weather and CAMS files are optional:
    when one is absent its columns are added as NaN and a warning is raised, so the backtest of the
    PM2.5-only baselines still runs. Returns exactly `synthetic.PANEL_COLS`, sorted by city, date.
    """
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    pm25_path = processed / PM25_FILE
    if not pm25_path.exists():
        raise FileNotFoundError(f"{pm25_path} not found: run `python -m thaiair.pm25` first to build it")

    frames = [_read_source(pm25_path, synthetic.PM25_COLS)]
    absent: list[str] = []
    for name, cols in OPTIONAL_FILES.items():
        path = processed / name
        if path.exists():
            frames.append(_read_source(path, cols))
        else:
            warnings.warn(f"{path} not found: its columns {cols[:3]}... will be NaN", stacklevel=2)
            absent.extend(cols)

    start = pd.Timestamp(config.DATA_START)
    frames = [df[df["date"] >= start] for df in frames]
    end = _last_date_with_data(frames)
    panel = _calendar(_ordered_cities(frames), start, end)
    for df in frames:
        panel = panel.merge(df, on=KEYS, how="left")
    for col in absent:
        panel[col] = float("nan")

    panel["n_stations"] = panel["n_stations"].fillna(0).astype(int)
    panel = panel[synthetic.PANEL_COLS].sort_values(KEYS).reset_index(drop=True)
    log.info("panel: %d rows, %d cities, %s .. %s", len(panel), panel["city"].nunique(), start.date(), end.date())
    return panel

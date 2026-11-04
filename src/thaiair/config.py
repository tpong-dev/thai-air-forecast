"""Shared settings: cities, stations, paths, dates and thresholds.

Every module imports from here so that the whole project agrees on one definition
of "a city", "a horizon" and "an unhealthy day".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("THAIAIR_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("THAIAIR_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TZ = "Asia/Bangkok"


@dataclass(frozen=True)
class City:
    key: str  # url-safe id, used in the API (?city=chiang_mai)
    name: str
    lat: float  # point used for weather and CAMS (city centre)
    lon: float
    stations: tuple[str, ...]  # Air4Thai station IDs (upper case, as in the PCD daily files)


# Stations chosen for long, near-complete records 2020-2026 (see docs/sources.md).
# City value for a day = median of the listed stations that reported that day.
CITIES: dict[str, City] = {
    c.key: c
    for c in [
        City("bangkok", "Bangkok", 13.7563, 100.5018, ("02T", "03T", "05T", "12T", "50T", "52T", "54T", "59T", "61T")),
        City("chiang_mai", "Chiang Mai", 18.7883, 98.9853, ("35T", "36T")),
        City("chiang_rai", "Chiang Rai", 19.9105, 99.8406, ("57T",)),
        City("nan", "Nan", 18.7838, 100.7784, ("67T",)),
        City("khon_kaen", "Khon Kaen", 16.4419, 102.8360, ("46T",)),
        City("nakhon_ratchasima", "Nakhon Ratchasima", 14.9799, 102.0978, ("47T",)),
        City("hat_yai", "Hat Yai", 7.0086, 100.4747, ("44T",)),
    ]
}

HORIZONS: tuple[int, ...] = (1, 2, 3)  # days ahead = 24 / 48 / 72 h

# Data window. PCD yearly files: 2019-01 .. 2026-06; Air4Thai history API adds Sep 2026 onwards.
DATA_START = "2019-01-01"
# Rolling-origin backtest: refit at the start of every month in this window, predict that month.
# Three burning seasons (2024, 2025, 2026) fall inside it; CAMS history starts Sep 2022.
TEST_START = "2023-10-01"
TEST_END = "2026-06-30"

# Burning / smog season in northern and central Thailand.
BURNING_MONTHS: tuple[int, ...] = (1, 2, 3, 4)

# Thai AQI PM2.5 bands (24 h mean, ug/m3), PCD 2023 revision.
# Upper edge of each band: very good, good, moderate, "starts to affect health", "affects health".
THAI_AQI_PM25_BANDS: tuple[tuple[float, str], ...] = (
    (15.0, "very good"),
    (25.0, "good"),
    (37.5, "moderate"),
    (75.0, "starts to affect health"),
    (float("inf"), "affects health"),
)
# An "unhealthy day" alert fires when the daily mean is above this (orange band or worse).
UNHEALTHY_PM25 = 37.5

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# Keep CPU use moderate: every model and library uses at most this many threads.
N_THREADS = int(os.environ.get("THAIAIR_THREADS", "2"))

HTTP_TIMEOUT = 60
USER_AGENT = "thai-air-forecast/0.1 (+https://github.com/tpong-dev/thai-air-forecast)"


def thai_aqi_band(pm25: float) -> str:
    for upper, label in THAI_AQI_PM25_BANDS:
        if pm25 <= upper:
            return label
    return THAI_AQI_PM25_BANDS[-1][1]

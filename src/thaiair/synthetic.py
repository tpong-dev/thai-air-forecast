"""Synthetic panel with the exact columns of the real one, for tests and CI (no network, no data files)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaiair import config

PM25_COLS = ["pm25", "n_stations"]
PAST_WEATHER_COLS = ["w_temp", "w_rh", "w_wind", "w_u", "w_v", "w_precip", "w_blh", "w_blh_min"]
FC_VARS = ["temp", "rh", "wind", "u", "v", "precip"]
FC_WEATHER_COLS = [f"f{h}_{v}" for h in config.HORIZONS for v in FC_VARS]
CAMS_COLS = ["cams_pm25"]
PANEL_COLS = ["city", "date", *PM25_COLS, *PAST_WEATHER_COLS, *FC_WEATHER_COLS, *CAMS_COLS]


def make_panel(
    start: str = "2021-01-01",
    end: str = "2024-06-30",
    cities: list[str] | None = None,
    seed: int = 0,
    missing_rate: float = 0.03,
) -> pd.DataFrame:
    """Seasonal PM2.5 driven by wind / boundary layer, with AR noise, outages and a noisy CAMS."""
    rng = np.random.default_rng(seed)
    cities = cities or list(config.CITIES)[:3]
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    frames = []
    for i, city in enumerate(cities):
        doy = dates.day_of_year.to_numpy()
        season = np.exp(-(((doy - 70) % 365) / 45.0) ** 2) + np.exp(-(((doy - 70 - 365) % 365) / 45.0) ** 2)
        wind = np.clip(2.5 + 1.5 * rng.standard_normal(n) - season, 0.2, None)
        blh = np.clip(900 - 400 * season + 200 * rng.standard_normal(n), 50, None)
        precip = np.where(rng.random(n) < 0.3 * (1 - season), rng.gamma(2, 5, n), 0.0)
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.7 * noise[t - 1] + rng.standard_normal()
        pm = 15 + (25 + 15 * i) * season + 3 * noise - 2 * wind - 0.3 * precip + 0.004 * (900 - blh)
        pm = np.clip(pm, 2, None)
        df = pd.DataFrame({"city": city, "date": dates, "pm25": pm, "n_stations": 1})
        df["w_temp"] = 28 + 3 * np.sin(2 * np.pi * doy / 365) + rng.standard_normal(n)
        df["w_rh"] = np.clip(70 - 15 * season + 5 * rng.standard_normal(n), 10, 100)
        df["w_wind"] = wind
        df["w_u"] = wind * np.cos(rng.random(n) * 2 * np.pi)
        df["w_v"] = wind * np.sin(rng.random(n) * 2 * np.pi)
        df["w_precip"] = precip
        df["w_blh"] = blh
        df["w_blh_min"] = blh * 0.2
        for h in config.HORIZONS:
            err = 0.3 * h
            df[f"f{h}_temp"] = df["w_temp"] + err * rng.standard_normal(n)
            df[f"f{h}_rh"] = df["w_rh"] + 3 * err * rng.standard_normal(n)
            df[f"f{h}_wind"] = np.clip(df["w_wind"] + err * rng.standard_normal(n), 0, None)
            df[f"f{h}_u"] = df["w_u"] + err * rng.standard_normal(n)
            df[f"f{h}_v"] = df["w_v"] + err * rng.standard_normal(n)
            df[f"f{h}_precip"] = np.clip(df["w_precip"] + 2 * err * rng.standard_normal(n), 0, None)
        df["cams_pm25"] = np.clip(1.3 * pm + 5 * rng.standard_normal(n), 0, None)
        out = rng.random(n) < missing_rate
        df.loc[out, "pm25"] = np.nan
        df.loc[out, "n_stations"] = 0
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["n_stations"] = panel["n_stations"].astype(int)
    return panel[PANEL_COLS].sort_values(["city", "date"]).reset_index(drop=True)

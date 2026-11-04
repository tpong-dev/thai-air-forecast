"""build_panel: complete calendar, exact columns, required vs optional sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thaiair import config, synthetic
from thaiair.dataset import build_panel


def _write_sources(panel: pd.DataFrame, folder: Path, skip: tuple[str, ...] = ()) -> None:
    """Split a synthetic panel back into the four processed files of the contract."""
    parts = {
        "pm25_daily.parquet": synthetic.PM25_COLS,
        "weather_past.parquet": synthetic.PAST_WEATHER_COLS,
        "weather_fc.parquet": synthetic.FC_WEATHER_COLS,
        "cams_daily.parquet": synthetic.CAMS_COLS,
    }
    for name, cols in parts.items():
        if name not in skip:
            panel[["city", "date", *cols]].to_parquet(folder / name, index=False)


@pytest.fixture
def source_panel() -> pd.DataFrame:
    return synthetic.make_panel(start="2019-01-01", end="2019-06-30", cities=["chiang_mai", "bangkok"], seed=3)


def test_round_trip_matches_source(tmp_path: Path, source_panel: pd.DataFrame) -> None:
    _write_sources(source_panel, tmp_path)
    panel = build_panel(tmp_path)
    assert list(panel.columns) == synthetic.PANEL_COLS
    expected = source_panel.sort_values(["city", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(panel, expected, check_dtype=False)


def test_complete_calendar_fills_gaps_with_nan(tmp_path: Path, source_panel: pd.DataFrame) -> None:
    dropped = source_panel.drop(index=[5, 6, 7])  # remove three pm25 rows entirely
    gap_dates = source_panel.loc[[5, 6, 7], "date"]
    _write_sources(source_panel, tmp_path)
    dropped[["city", "date", *synthetic.PM25_COLS]].to_parquet(tmp_path / "pm25_daily.parquet", index=False)
    panel = build_panel(tmp_path)
    assert len(panel) == len(source_panel)
    for city, g in panel.groupby("city"):
        assert g["date"].diff().dropna().eq(pd.Timedelta(days=1)).all(), city
    rows = panel[(panel["city"] == source_panel.loc[5, "city"]) & panel["date"].isin(gap_dates)]
    assert rows["pm25"].isna().all() and (rows["n_stations"] == 0).all()
    assert rows["w_wind"].notna().all()  # other sources still joined on those days


def test_calendar_starts_at_data_start(tmp_path: Path) -> None:
    late = synthetic.make_panel(start="2019-03-01", end="2019-03-31", cities=["nan"])
    _write_sources(late, tmp_path)
    panel = build_panel(tmp_path)
    assert panel["date"].min() == pd.Timestamp(config.DATA_START)
    assert panel["date"].max() == pd.Timestamp("2019-03-31")
    assert panel.loc[panel["date"] < "2019-03-01", "pm25"].isna().all()


def test_optional_sources_become_nan_columns(tmp_path: Path, source_panel: pd.DataFrame) -> None:
    _write_sources(source_panel, tmp_path, skip=("weather_fc.parquet", "cams_daily.parquet"))
    with pytest.warns(UserWarning, match="not found"):
        panel = build_panel(tmp_path)
    assert list(panel.columns) == synthetic.PANEL_COLS
    assert panel[[*synthetic.FC_WEATHER_COLS, *synthetic.CAMS_COLS]].isna().all().all()
    assert panel["w_blh"].notna().any()


def test_missing_pm25_is_a_clear_error(tmp_path: Path, source_panel: pd.DataFrame) -> None:
    _write_sources(source_panel, tmp_path, skip=("pm25_daily.parquet",))
    with pytest.raises(FileNotFoundError, match="thaiair.pm25"):
        build_panel(tmp_path)


def test_timezone_aware_dates_and_duplicates(tmp_path: Path, source_panel: pd.DataFrame) -> None:
    _write_sources(source_panel, tmp_path)
    cams = source_panel[["city", "date", "cams_pm25"]].copy()
    cams["date"] = cams["date"].dt.tz_localize(config.TZ)
    cams.to_parquet(tmp_path / "cams_daily.parquet", index=False)
    panel = build_panel(tmp_path)
    assert np.allclose(panel["cams_pm25"], source_panel.sort_values(["city", "date"])["cams_pm25"])

    pd.concat([cams, cams.head(1)]).to_parquet(tmp_path / "cams_daily.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        build_panel(tmp_path)

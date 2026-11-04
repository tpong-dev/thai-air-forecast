"""Tests for thaiair.pm25. No network and no data files: fixtures are built in tmp_path, HTTP is mocked."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from thaiair import config, pm25


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any real HTTP call fails the test; the trust-store adapter is not mounted."""
    monkeypatch.setattr(pm25, "_use_os_trust_store", lambda: None)

    def _fail(*args, **kwargs):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(pm25.http, "get_json", _fail)
    monkeypatch.setattr(pm25.http, "get", _fail)


def _write_quirky_xlsx(path: Path) -> None:
    """A tiny yearly file with the quirks seen in the real PCD files."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DATA"
    ws.append(["Some title row"])  # header shifted down by one row
    ws.append(["Date", " 35t", "36T", "36T", None])  # odd case/space, duplicated column, empty header
    ws.append([datetime(2021, 6, 28), 10, "N/A", 7, None])
    ws.append([datetime(2021, 6, 30), "n/a", 20.5, None, None])  # typo: should be 2021-06-29
    ws.append([datetime(2021, 6, 30), "12", 21, None, None])  # number stored as text
    ws.append([datetime(2020, 12, 31), 99, 99, None, None])  # row from another year
    ws.append([None, None, None, None, None])
    ws.append(["หมายเหตุ : N/A ไม่มีข้อมูล", None, None, None, None])  # Thai footnote
    wb.create_sheet("stations").append(["รหัสสถานี", "ชื่อสถานี"])
    wb.save(path)


def test_read_pcd_xlsx_handles_quirks(tmp_path: Path) -> None:
    path = tmp_path / "2021.xlsx"
    _write_quirky_xlsx(path)
    parsed = pm25.read_pcd_xlsx(path)
    d = parsed.data.set_index(["station", "date"])["pm25"]

    assert set(parsed.data["station"]) == {"35T", "36T"}
    assert parsed.data["date"].dtype == "datetime64[ns]"
    assert parsed.data["date"].min() == pd.Timestamp("2021-06-28")  # 2020 row dropped
    assert not parsed.data.duplicated(["station", "date"]).any()
    assert np.isnan(d[("35T", pd.Timestamp("2021-06-29"))])  # relabelled date, "n/a" -> NaN
    assert d[("35T", pd.Timestamp("2021-06-30"))] == 12.0  # text number parsed
    assert d[("36T", pd.Timestamp("2021-06-28"))] == 7.0  # duplicated column: first non-missing kept
    assert parsed.n_markers == 2
    assert any("relabelled" in n for n in parsed.notes)
    assert any("outside 2021" in n for n in parsed.notes)


def test_fix_duplicate_dates_drops_when_ambiguous() -> None:
    dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-03", "2021-01-02"]))
    fixed, notes = pm25.fix_duplicate_dates(dates)
    assert list(fixed) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    assert "dropped" in notes[0]


def _hourly(station: str, day: str, values: list[float | None]) -> pd.DataFrame:
    times = pd.date_range(day, periods=len(values), freq="h")
    return pd.DataFrame({"station": station, "time": times, "pm25": pd.array(values, dtype=float)})


def test_hourly_to_daily_needs_18_valid_hours() -> None:
    ok = _hourly("35T", "2026-10-01", [10.0] * 18 + [None] * 6)
    short = _hourly("35T", "2026-10-02", [10.0] * 17 + [None] * 7)
    junk = _hourly("35T", "2026-10-03", [20.0] * 18 + [-1.0, 9999.0] + [None] * 4)
    out = pm25.hourly_to_daily(pd.concat([ok, short, junk]))
    assert list(out["date"]) == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-03")]
    assert list(out["pm25"]) == [10.0, 20.0]  # -1 and 9999 ignored
    assert list(out["n_hours"]) == [18, 18]


def _station_series(values: list[float], station: str = "35T", start: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame({"station": station, "date": pd.date_range(start, periods=len(values)), "pm25": values})


def test_cleaning_rules() -> None:
    stuck = [10.0, 11.0] + [14.0] * 7 + [12.0]
    six_same = [14.0] * 6 + [15.0]
    spike = [20.0, 25.0, 400.0, 30.0, 28.0]
    burning = [60.0, 120.0, 250.0, 180.0, 90.0]  # real smoke episode: must survive
    bad = [-5.0, 2000.0, 15.0]
    df = pd.concat([
        _station_series(stuck, "A"), _station_series(six_same, "B"), _station_series(spike, "C"),
        _station_series(burning, "D"), _station_series(bad, "E"),
    ])
    out, counts = pm25.clean_station_daily(df)
    assert counts == {"out_of_range": 2, "stuck": 7, "spike": 1}
    by = {s: g["pm25"].tolist() for s, g in out.groupby("station")}
    assert np.isnan(by["A"][2:9]).all() and by["A"][9] == 12.0
    assert by["B"] == six_same
    assert np.isnan(by["C"][2]) and by["C"][3] == 30.0
    assert by["D"] == burning
    assert np.isnan(by["E"][:2]).all()


def test_spike_needs_both_neighbours() -> None:
    df = _station_series([400.0, 20.0, 20.0])  # first day has no left neighbour
    out, counts = pm25.clean_station_daily(df)
    assert counts["spike"] == 0 and out["pm25"].iloc[0] == 400.0


def test_combine_sources_prefers_pcd() -> None:
    pcd = _station_series([10.0, np.nan], "35T")
    hourly = _station_series([99.0, 99.0, 30.0], "35T")
    out = pm25.combine_sources(pcd, hourly).set_index("date")
    assert list(out["source"]) == ["pcd_daily", "pcd_daily", "a4t_hourly"]
    assert out["pm25"].iloc[0] == 10.0 and np.isnan(out["pm25"].iloc[1]) and out["pm25"].iloc[2] == 30.0


def test_city_daily_median_and_contract() -> None:
    bkk = config.CITIES["bangkok"].stations
    rows = [
        (bkk[0], "2019-01-01", 10.0), (bkk[1], "2019-01-01", 20.0), (bkk[2], "2019-01-01", 90.0),
        (bkk[0], "2019-01-03", np.nan),
        ("35T", "2019-01-02", 40.0), ("36T", "2019-01-02", 50.0),
        ("99T", "2019-01-02", 500.0),  # not a configured station: ignored
    ]
    sd = pd.DataFrame(rows, columns=["station", "date", "pm25"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    out = pm25.city_daily(sd, start="2019-01-01", end="2019-01-03")

    assert list(out.columns) == ["city", "date", "pm25", "n_stations"]
    assert out["date"].dtype == "datetime64[ns]"
    assert out["n_stations"].dtype == "int64" and out["pm25"].dtype == "float64"
    assert len(out) == 3 * len(config.CITIES)  # full calendar for every city
    assert not out.duplicated(["city", "date"]).any()
    assert out.equals(out.sort_values(["city", "date"]).reset_index(drop=True))
    b = out[out["city"] == "bangkok"].set_index("date")
    assert b.loc["2019-01-01", "pm25"] == 20.0 and b.loc["2019-01-01", "n_stations"] == 3
    assert np.isnan(b.loc["2019-01-03", "pm25"]) and b.loc["2019-01-03", "n_stations"] == 0
    cm = out[out["city"] == "chiang_mai"].set_index("date")
    assert cm.loc["2019-01-02", "pm25"] == 45.0 and cm.loc["2019-01-02", "n_stations"] == 2


def _hourly_doc(station: str, day: str, value: float | None, hours: int = 24) -> dict:
    data = [{"DATETIMEDATA": f"{day} {h:02d}:00:00", "PM25": value} for h in range(hours)]
    return {"result": "OK", "error": "", "stations": [{"stationID": station.lower(), "data": data}]}


def test_recent_daily_with_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_get_json(url, params=None, cache=None, pause=0.0):
        calls.append(params)
        sid = params["stationID"]
        docs = [_hourly_doc(sid, "2026-11-01", 30.0 if sid == "35t" else 40.0),
                _hourly_doc(sid, "2026-11-02", 10.0, hours=12),  # incomplete day
                _hourly_doc(sid, "2026-11-04", 99.0)]  # "today": must be ignored
        return {"result": "OK", "stations": [{"stationID": sid, "data": [r for d in docs for r in d["stations"][0]["data"]]}]}

    monkeypatch.setattr(pm25.http, "get_json", fake_get_json)
    out = pm25.recent_daily("chiang_mai", days=3, today=date(2026, 11, 4))

    assert {c["stationID"] for c in calls} == {"35t", "36t"}
    assert all(c["sdate"] == "2026-11-01" and c["edate"] == "2026-11-03" for c in calls)
    assert list(out["date"]) == list(pd.date_range("2026-11-01", "2026-11-03"))
    day = out.set_index("date")
    assert day.loc["2026-11-01", "pm25"] == 35.0 and day.loc["2026-11-01", "n_stations"] == 2
    assert np.isnan(day.loc["2026-11-02", "pm25"]) and day.loc["2026-11-02", "n_stations"] == 0
    assert list(out.columns) == ["city", "date", "pm25", "n_stations"]


def test_fetch_hourly_caches_complete_months(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_get_json(url, params=None, cache=None, pause=0.0):
        calls.append(params["sdate"])
        return _hourly_doc(params["stationID"], params["sdate"], 12.0)

    monkeypatch.setattr(pm25.http, "get_json", fake_get_json)
    monkeypatch.setattr(pm25, "bangkok_today", lambda: date(2026, 11, 4))
    for _ in range(2):
        pm25.fetch_hourly("35T", date(2026, 10, 1), date(2026, 11, 3), raw_dir=tmp_path)
    # October is complete -> fetched once; November is still open -> fetched on every run.
    assert calls == ["2026-10-01", "2026-11-01", "2026-11-01"]
    assert (tmp_path / "35T_2026-10.json").exists()


def test_parse_latest_marks_minus_one_missing() -> None:
    doc = {"stations": [
        {"stationID": "35t", "nameEN": "Chang Phueak", "areaEN": "Chiang Mai", "lat": "18.8", "long": "98.97",
         "AQILast": {"date": "2026-11-04", "time": "19:00", "PM25": {"value": "21.5"}}},
        {"stationID": "36t", "nameEN": "Si Phum", "areaEN": "Chiang Mai", "lat": "18.79", "long": "98.99",
         "AQILast": {"date": "2026-11-04", "time": "19:00", "PM25": {"value": "-1"}}},
    ]}
    out = pm25.parse_latest(doc)
    assert list(out.columns) == ["station", "name", "area", "lat", "lon", "pm25", "time"]
    assert list(out["station"]) == ["35T", "36T"]
    assert out["pm25"].iloc[0] == 21.5 and np.isnan(out["pm25"].iloc[1])
    assert out["time"].iloc[0] == pd.Timestamp("2026-11-04 19:00")


def test_build_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    (raw / "pcd_daily").mkdir(parents=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "35T", "36T", "02T"])
    for day in range(1, 4):
        ws.append([datetime(2026, 6, 27 + day), 10 * day, 20 * day, "N/A"])
    wb.save(raw / "pcd_daily" / "2026.xlsx")

    def fake_get_json(url, params=None, cache=None, pause=0.0):
        return _hourly_doc(params["stationID"], "2026-07-02", 5.0) if params["sdate"] == "2026-07-01" else {"result": "OK", "stations": []}

    monkeypatch.setattr(pm25.http, "get_json", fake_get_json)
    monkeypatch.setattr(pm25, "bangkok_today", lambda: date(2026, 7, 4))
    monkeypatch.setattr(config, "DATA_START", "2026-06-28")
    res = pm25.build(raw, processed, today=date(2026, 7, 4), years=[2026])

    daily = pd.read_parquet(processed / "pm25_daily.parquet")
    station = pd.read_parquet(processed / "pm25_station_daily.parquet")
    assert daily["date"].dtype == "datetime64[ns]" and daily["n_stations"].dtype == "int64"
    assert daily["date"].min() == pd.Timestamp("2026-06-28") and daily["date"].max() == pd.Timestamp("2026-07-02")
    cm = daily[daily["city"] == "chiang_mai"].set_index("date")["pm25"]
    assert cm["2026-06-28"] == 15.0 and np.isnan(cm["2026-07-01"]) and cm["2026-07-02"] == 5.0
    assert set(station["source"]) == {"pcd_daily", "a4t_hourly"}
    assert list(station.columns) == ["station", "date", "pm25", "source"]
    assert res.counts["pcd_missing"] == 3

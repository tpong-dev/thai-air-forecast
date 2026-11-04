"""Tests for thaiair.report: a fake predictions frame in, results.md and three PNGs out (no network, no data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thaiair import report

SECTIONS = ["headline", "by_horizon", "by_city", "by_season", "alerts", "coverage", "skill", "cams_wins"]


def fake_preds(models: tuple[str, ...] = ("persistence", "seasonal", "cams", "cams_bc", "lgbm"), seed: int = 0) -> pd.DataFrame:
    """Long predictions over one burning season for three cities, with a smog peak in March."""
    rng = np.random.default_rng(seed)
    issue = pd.date_range("2023-10-01", "2024-06-27", freq="D")
    doy = issue.day_of_year.to_numpy()
    rows = []
    for i, city in enumerate(["bangkok", "chiang_mai", "nan"]):
        season = np.exp(-(((doy - 75) % 365) / 40.0) ** 2)
        for h in (1, 2, 3):
            target = issue + pd.Timedelta(days=h)
            y = 15 + (30 + 20 * i) * season + 4 * rng.standard_normal(len(issue))
            y[rng.random(len(issue)) < 0.02] = np.nan  # outages
            for j, m in enumerate(models):
                noise = (2 + j + h) * rng.standard_normal(len(issue))
                p50 = np.clip(y + noise + (5 if m == "cams" else 0), 0, None)
                has_int = m in ("lgbm", "deep")
                rows.append(pd.DataFrame({
                    "model": m, "city": city, "horizon": h, "issue_date": issue, "target_date": target, "y": y,
                    "p10": p50 - 6 if has_int else np.nan, "p50": p50, "p90": p50 + 6 if has_int else np.nan,
                }))
    df = pd.concat(rows, ignore_index=True)
    # One model misses a few rows; pairing must drop those rows for every model.
    gap = (df["model"] == models[-1]) & (df["issue_date"] < "2023-10-05")
    return df[~gap].reset_index(drop=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Path]]:
    pytest.importorskip("matplotlib")
    out = tmp_path_factory.mktemp("reports")
    return out, report.build_report(fake_preds(), out)


def test_writes_results_and_figures(built: tuple[Path, dict[str, Path]]) -> None:
    out, paths = built
    text = (out / "results.md").read_text(encoding="utf-8")
    blocks = report.result_blocks(text)
    for name in SECTIONS:
        assert name in blocks and blocks[name].strip(), name
    for name in ("smog_episode", "error_by_horizon", "skill_by_city"):
        png = out / "figures" / f"{name}.png"
        assert png.exists() and png.stat().st_size > 10_000, name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert paths[name] == png


def test_report_content(built: tuple[Path, dict[str, Path]]) -> None:
    text = (built[0] / "results.md").read_text(encoding="utf-8")
    assert "LightGBM" in text and "Chiang Mai" in text
    assert "stitched" in text  # the CAMS lead-time caveat is printed next to the numbers
    assert "CAMS†" in report.result_blocks(text)["skill"]
    assert "CAMS wins in" in report.result_blocks(text)["cams_wins"]
    assert "h=3 burning" in text


def test_paired_keeps_only_complete_rows() -> None:
    df = report.prepare(fake_preds())
    paired = report.paired_local(df)
    per_key = paired.groupby(report.KEYS)["model"].nunique()
    assert (per_key == df["model"].nunique()).all()
    assert paired["y"].notna().all()
    assert paired["issue_date"].min() >= pd.Timestamp("2023-10-05")


def test_missing_cams_is_an_error() -> None:
    df = report.prepare(fake_preds(models=("persistence", "lgbm")))
    with pytest.raises(ValueError, match="cams"):
        report.paired_local(df)


def test_best_model_falls_back_when_lgbm_absent() -> None:
    rep = report.Report(fake_preds(models=("persistence", "cams", "deep")), report._Metrics.load())
    assert rep.best == "deep"
    rep = report.Report(fake_preds(models=("persistence", "cams")), report._Metrics.load())
    assert rep.best == "persistence"


def test_alert_scores_by_hand() -> None:
    df = pd.DataFrame({
        "model": ["m"] * 5,
        "y": [10.0, 40.0, 50.0, 60.0, 20.0],
        "p50": [45.0, 50.0, 30.0, 70.0, 10.0],
    })
    a = report.alerts_local(df, ["model"]).iloc[0]
    assert a["events"] == 3 and a["alerts"] == 3
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(2 / 3)
    assert a["f1"] == pytest.approx(2 / 3)


def test_coverage_by_hand() -> None:
    df = pd.DataFrame({"model": ["m"] * 4, "y": [1.0, 5.0, 9.0, 20.0], "p10": [0.0] * 4, "p90": [10.0] * 4})
    c = report.coverage_local(df, ["model"]).iloc[0]
    assert c["coverage"] == pytest.approx(0.75)
    assert c["width"] == pytest.approx(10.0)


def test_episode_window_finds_the_peak() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    s = pd.Series(10.0, index=idx)
    s.iloc[100:160] = 80.0
    start, end = report.episode_window(s, days=60)
    assert start == idx[100] and end == idx[159]


def test_fill_readme_only_touches_blocks(tmp_path: Path) -> None:
    results = tmp_path / "results.md"
    results.write_text("<!-- RESULTS:by_horizon -->\n| a |\n![x](figures/x.png)\n<!-- /RESULTS -->\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "Intro\n<!-- RESULTS:by_horizon -->\nold\n<!-- /RESULTS -->\nTODO(lead) keep me\n"
        "<!-- RESULTS:unknown -->\nkeep\n<!-- /RESULTS -->\n",
        encoding="utf-8",
    )
    assert report.fill_readme(readme, results) == ["by_horizon"]
    text = readme.read_text(encoding="utf-8")
    assert "| a |" in text and "old" not in text
    assert "](reports/figures/x.png)" in text
    assert "TODO(lead) keep me" in text and "keep\n" in text

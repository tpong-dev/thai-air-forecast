"""Metrics on a tiny hand-made prediction table (numbers checkable by hand)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaiair import metrics


def _preds() -> pd.DataFrame:
    """Two models, one city, horizon 1, four days. CAMS is missing on the first day."""
    dates = pd.date_range("2024-03-30", periods=4)  # two burning-season days (March), two "rest" (April)
    y = [10.0, 40.0, 50.0, 20.0]
    base = {"city": "bangkok", "horizon": 1, "issue_date": dates - pd.Timedelta(days=1), "target_date": dates, "y": y}
    a = pd.DataFrame(
        {
            **base,
            "model": "a",
            "p50": [12.0, 38.0, 30.0, 20.0],
            "p10": [5.0, 30.0, 25.0, 10.0],
            "p90": [15.0, 45.0, 45.0, 30.0],
        }
    )
    c = pd.DataFrame({**base, "model": "cams", "p50": [np.nan, 50.0, 60.0, 30.0], "p10": np.nan, "p90": np.nan})
    return pd.concat([a, c], ignore_index=True)


def test_season() -> None:
    assert metrics.season(pd.Timestamp("2024-02-01")) == "burning"
    assert metrics.season(pd.Timestamp("2024-07-01")) == "rest"
    labels = metrics.season(pd.Series(pd.to_datetime(["2024-04-30", "2024-05-01"])))
    assert labels.tolist() == ["burning", "rest"]


def test_score_table() -> None:
    t = metrics.score_table(_preds(), by=["horizon"]).set_index("model")
    assert t.loc["a", "n"] == 4 and t.loc["cams", "n"] == 3
    assert t.loc["a", "mae"] == pytest.approx((2 + 2 + 20 + 0) / 4)
    assert t.loc["a", "rmse"] == pytest.approx(np.sqrt((4 + 4 + 400) / 4))
    assert t.loc["a", "bias"] == pytest.approx((2 - 2 - 20 + 0) / 4)


def test_score_table_by_season() -> None:
    t = metrics.score_table(_preds(), by=["season"])
    a = t[t["model"] == "a"].set_index("season")
    assert a.loc["burning", "n"] == 4  # 30 Mar - 2 Apr are all in BURNING_MONTHS (1-4)


def test_paired_restricts_to_common_rows() -> None:
    p = metrics.paired(_preds(), ["a", "cams"])
    assert len(p) == 6 and set(p["model"]) == {"a", "cams"}
    assert p["target_date"].min() == pd.Timestamp("2024-03-31")
    assert list(p.columns) == list(_preds().columns)


def test_skill_on_paired_rows() -> None:
    s = metrics.skill(_preds(), ref="cams").set_index("model")
    mae_a, mae_c = (2 + 20 + 0) / 3, (10 + 10 + 10) / 3
    assert s.loc["a", "n"] == 3
    assert s.loc["a", "mae_model"] == pytest.approx(mae_a)
    assert s.loc["a", "skill"] == pytest.approx(1 - mae_a / mae_c)
    with pytest.raises(ValueError):
        metrics.skill(_preds(), ref="missing")


def test_alert_scores() -> None:
    t = metrics.alert_scores(_preds(), threshold=37.5).set_index("model")
    # truth: events on days 2 and 3. model a alerts on day 2 only; cams alerts on days 2 and 3 (day 1 NaN).
    assert t.loc["a", "n_true_events"] == 2 and t.loc["a", "n_alerts"] == 1
    assert t.loc["a", "precision"] == 1.0 and t.loc["a", "recall"] == 0.5
    assert t.loc["a", "f1"] == pytest.approx(2 / 3)
    assert t.loc["cams", "recall"] == 1.0 and t.loc["cams", "f1"] == 1.0


def test_alert_scores_no_alerts_gives_nan_precision() -> None:
    p = _preds()
    p["p50"] = 1.0
    t = metrics.alert_scores(p).set_index("model")
    assert np.isnan(t.loc["a", "precision"]) and t.loc["a", "recall"] == 0.0


def test_coverage_skips_models_without_intervals() -> None:
    t = metrics.coverage(_preds(), by=["horizon"])
    assert t["model"].tolist() == ["a"]
    assert t.loc[0, "coverage"] == pytest.approx(3 / 4)  # day 3: 50 is above p90 = 45
    assert t.loc[0, "mean_width"] == pytest.approx((10 + 15 + 20 + 20) / 4)

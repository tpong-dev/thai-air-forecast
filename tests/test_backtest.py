"""Backtest harness: expanding window without look-ahead, nothing silently dropped."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaiair import backtest, synthetic
from thaiair.baselines import Cams, Persistence
from thaiair.models.base import PRED_COLS

TEST_START, TEST_END = "2024-01-01", "2024-03-31"


class Recorder:
    """Fake forecaster that remembers every training set it was given."""

    name = "recorder"
    fits: list[dict] = []

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def fit(self, train: pd.DataFrame) -> Recorder:
        Recorder.fits.append(
            {
                "horizon": self.horizon,
                "max_target": train["target_date"].max(),
                "max_issue": train["issue_date"].max(),
                "y_nan": int(train["y"].isna().sum()),
                "n": len(train),
            }
        )
        self.fold_start = train["target_date"].max() + pd.Timedelta(days=1)
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        assert (test["issue_date"] >= self.fold_start).all()
        return pd.DataFrame({"p10": 0.0, "p50": 1.0, "p90": 2.0}, index=test.index)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic.make_panel(start="2022-06-01", end="2024-04-30", seed=2)


def test_fold_starts() -> None:
    assert (
        backtest.fold_starts("2024-01-15", "2024-03-31")
        == pd.to_datetime(["2024-01-15", "2024-02-01", "2024-03-01"]).tolist()
    )
    assert len(backtest.fold_starts("2024-01-01", "2024-12-31", "QS")) == 4


def test_no_training_label_after_fold_start(panel: pd.DataFrame) -> None:
    Recorder.fits = []
    backtest.run(panel, {"rec": Recorder}, test_start=TEST_START, test_end=TEST_END)
    starts = backtest.fold_starts(TEST_START, TEST_END)
    assert len(Recorder.fits) == 3 * len(starts)
    for fit, start in zip(Recorder.fits, starts * 3):
        assert fit["max_target"] < start
        assert fit["y_nan"] == 0
    # Expanding window: training sets grow fold after fold.
    for h in (1, 2, 3):
        sizes = [f["n"] for f in Recorder.fits if f["horizon"] == h]
        assert sizes == sorted(sizes)


def test_output_shape_and_nan_rows_kept(panel: pd.DataFrame) -> None:
    p = panel.copy()
    p.loc[p["date"] == "2024-02-10", "pm25"] = np.nan
    p.loc[p["date"] < "2024-02-01", "cams_pm25"] = np.nan
    out = backtest.run(
        p, {"persistence": lambda h: Persistence(), "cams": lambda h: Cams()}, test_start=TEST_START, test_end=TEST_END
    )
    assert list(out.columns) == backtest.OUTPUT_COLS
    n_days = len(pd.date_range(TEST_START, TEST_END))
    expected = n_days * p["city"].nunique() * 3
    counts = out.groupby("model").size()
    assert (counts == expected).all()
    assert out["issue_date"].between(TEST_START, TEST_END).all()
    assert out.loc[out["target_date"] == "2024-02-10", "y"].isna().all()
    cams = out[out["model"] == "cams"]
    assert cams.loc[cams["target_date"] < "2024-02-01", "p50"].isna().all()
    assert cams.loc[cams["target_date"] >= "2024-02-01", "p50"].notna().all()


def test_per_model_refit_frequency(panel: pd.DataFrame) -> None:
    Recorder.fits = []
    backtest.run(panel, {"rec": Recorder}, horizons=(1,), test_start=TEST_START, test_end=TEST_END, refit={"rec": "QS"})
    assert len(Recorder.fits) == 1
    Recorder.fits = []
    backtest.run(
        panel, {"rec": Recorder}, horizons=(1,), test_start=TEST_START, test_end=TEST_END, refit={"other": "QS"}
    )
    assert len(Recorder.fits) == 3


def test_deterministic(panel: pd.DataFrame) -> None:
    factories = {"persistence": lambda h: Persistence()}
    a = backtest.run(panel, factories, horizons=(2,), test_start=TEST_START, test_end=TEST_END)
    b = backtest.run(panel, factories, horizons=(2,), test_start=TEST_START, test_end=TEST_END)
    pd.testing.assert_frame_equal(a, b)


def test_bad_predict_index_fails_loudly(panel: pd.DataFrame) -> None:
    class Broken(Persistence):
        def predict(self, test: pd.DataFrame) -> pd.DataFrame:
            return super().predict(test).reset_index(drop=True)[PRED_COLS]

    with pytest.raises(ValueError, match="same index"):
        backtest.run(panel, {"broken": lambda h: Broken()}, horizons=(1,), test_start=TEST_START, test_end=TEST_END)

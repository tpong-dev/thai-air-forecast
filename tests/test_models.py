"""LightGBM and deep quantile models: output contract, sanity vs persistence, calibration, saving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaiair import synthetic
from thaiair.features import make_features
from thaiair.models import registry
from thaiair.models.base import PRED_COLS, Forecaster
from thaiair.models.deep import DeepQuantile
from thaiair.models.lgbm import LightGBMQuantile

SPLIT = "2024-01-01"  # test = Jan-Jun 2024, includes a synthetic burning season
FAST_DEEP = {"max_epochs": 15}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic.make_panel()


@pytest.fixture(scope="module")
def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = make_features(panel, 1)
    feats = feats[feats["y"].notna()]
    is_test = feats["target_date"] >= SPLIT
    return feats[~is_test], feats[is_test]


@pytest.fixture(scope="module")
def lgbm(split: tuple[pd.DataFrame, pd.DataFrame]) -> LightGBMQuantile:
    return LightGBMQuantile(1).fit(split[0])


@pytest.fixture(scope="module")
def deep(split: tuple[pd.DataFrame, pd.DataFrame]) -> DeepQuantile:
    return DeepQuantile(1, **FAST_DEEP).fit(split[0])


def _check_contract(pred: pd.DataFrame, test: pd.DataFrame) -> None:
    assert list(pred.columns) == PRED_COLS
    assert pred.index.equals(test.index)
    assert pred.notna().all().all()
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert (pred >= 0).all().all()


def _persistence_mae(panel: pd.DataFrame, test: pd.DataFrame) -> float:
    """Naive forecast: tomorrow = today's observed value (from the panel, not from a feature)."""
    today = panel.set_index(["city", "date"])["pm25"]
    naive = today.reindex(pd.MultiIndex.from_arrays([test["city"], test["issue_date"]])).to_numpy()
    return float(np.nanmean(np.abs(test["y"].to_numpy() - naive)))


def _small_frame(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Hand-made table (string city codes, explicit columns): the models must not need features.py."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    frames = []
    for city in ("a", "b"):
        x = rng.normal(size=n_days)
        frames.append(
            pd.DataFrame(
                {
                    "city": city,
                    "issue_date": dates,
                    "target_date": dates + pd.Timedelta(days=1),
                    "x1": x,
                    "x2": rng.normal(size=n_days),
                    "city_code": city,
                    "y": np.exp(2 + 0.5 * x + (city == "b")) + rng.gamma(1, 1, n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- LightGBM


def test_lgbm_output_contract(lgbm: LightGBMQuantile, split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    test = split[1]
    assert isinstance(lgbm, Forecaster) and lgbm.name == "lgbm"
    _check_contract(lgbm.predict(test), test)


def test_lgbm_beats_persistence(lgbm, split, panel) -> None:
    test = split[1]
    mae = float(np.mean(np.abs(test["y"] - lgbm.predict(test)["p50"])))
    assert mae < _persistence_mae(panel, test)


def test_lgbm_calibrated_coverage_near_80(lgbm, split) -> None:
    test = split[1]
    pred = lgbm.predict(test)
    coverage = float(((test["y"] >= pred["p10"]) & (test["y"] <= pred["p90"])).mean())
    assert lgbm.calibration_["n_cal"] > 0
    assert 0.65 <= coverage <= 0.95


def test_lgbm_calibration_switch(split) -> None:
    train, test = split
    raw = LightGBMQuantile(1, params={"n_estimators": 100}, calibrate=False).fit(train)
    assert raw.calibration_["n_cal"] == 0 and raw.calibration_["offset_lo"] == 0.0
    _check_contract(raw.predict(test), test)


def test_lgbm_feature_importance(lgbm) -> None:
    imp = lgbm.feature_importance()
    assert set(imp.index) == set(lgbm.feature_cols_)
    assert (imp >= 0).all() and imp.is_monotonic_decreasing


def test_lgbm_generic_frame_and_unseen_city() -> None:
    df = _small_frame()
    model = LightGBMQuantile(1, feature_cols=["x1", "x2", "city_code"], params={"n_estimators": 50}).fit(df)
    new = df.tail(5).assign(city_code="zz")  # a city the model never saw
    _check_contract(model.predict(new), new)


# --------------------------------------------------------------------------- deep


def test_deep_output_contract(deep: DeepQuantile, split) -> None:
    test = split[1]
    assert isinstance(deep, Forecaster) and deep.name == "deep"
    _check_contract(deep.predict(test), test)


def test_deep_is_deterministic(split) -> None:
    train, test = split
    a = DeepQuantile(1, seed=3, max_epochs=3).fit(train).predict(test)
    b = DeepQuantile(1, seed=3, max_epochs=3).fit(train).predict(test)
    pd.testing.assert_frame_equal(a, b)


def test_deep_survives_nan_inputs() -> None:
    df = _small_frame()
    df.loc[df.index[::7], "x1"] = np.nan  # gaps in training -> a missing-indicator column
    model = DeepQuantile(1, feature_cols=["x1", "x2", "city_code"], max_epochs=5).fit(df)
    assert model.missing_cols_ == ["x1"]
    new = df.tail(6).copy()
    new.loc[new.index[:3], ["x1", "x2"]] = np.nan  # x2 never missing in training: median only
    new.loc[new.index[3], "city_code"] = "zz"
    _check_contract(model.predict(new), new)


# --------------------------------------------------------------------------- registry


@pytest.mark.parametrize("which", ["lgbm", "deep"])
def test_save_load_round_trip(which, lgbm, deep, split, tmp_path) -> None:
    model = {"lgbm": lgbm, "deep": deep}[which]
    test = split[1]
    path = registry.save(model, which, 1, tmp_path)
    assert path.name == f"{which}_h1.joblib"
    loaded = registry.load(which, 1, tmp_path)
    pd.testing.assert_frame_equal(loaded.predict(test), model.predict(test))
    assert registry.available(tmp_path) == [(which, 1)]


def test_registry_missing_and_unknown(tmp_path) -> None:
    assert registry.available(tmp_path / "nowhere") == []
    assert registry.read_manifest(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        registry.load("lgbm", 1, tmp_path)
    with pytest.raises(ValueError):
        registry.make_model("prophet", 1)


def test_manifest_round_trip(tmp_path) -> None:
    entry: registry.ManifestEntry = {
        "name": "lgbm",
        "horizon": 2,
        "file": "lgbm_h2.joblib",
        "feature_columns": ["a", "b"],
        "n_train_rows": 10,
    }
    registry.write_manifest([entry], ("2021-01-02", "2024-06-30"), tmp_path)
    manifest = registry.read_manifest(tmp_path)
    assert manifest is not None
    assert manifest["names"] == ["lgbm"] and manifest["horizons"] == [2]
    assert manifest["feature_columns"] == {"2": ["a", "b"]}
    assert manifest["train_period"] == {"start": "2021-01-02", "end": "2024-06-30"}
    assert manifest["library_versions"]["lightgbm"]

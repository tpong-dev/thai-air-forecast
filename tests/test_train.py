"""train_final writes one model per (name, horizon), the manifest and LightGBM importances."""

from __future__ import annotations

import pandas as pd

from thaiair import synthetic
from thaiair.features import feature_columns, make_features
from thaiair.models import registry
from thaiair.train import train_final


def test_train_final_writes_models_manifest_and_importance(tmp_path) -> None:
    panel = synthetic.make_panel(start="2022-01-01", end="2023-12-31", cities=["bangkok", "chiang_mai"])
    models_dir, metrics_dir = tmp_path / "models", tmp_path / "metrics"

    entries = train_final(
        panel, names=("lgbm", "deep"), horizons=(1, 2), models_dir=models_dir, metrics_dir=metrics_dir
    )

    assert len(entries) == 4
    assert registry.available(models_dir) == [("deep", 1), ("deep", 2), ("lgbm", 1), ("lgbm", 2)]
    manifest = registry.read_manifest(models_dir)
    assert manifest is not None
    assert manifest["names"] == ["deep", "lgbm"] and manifest["horizons"] == [1, 2]
    assert manifest["feature_columns"]["2"] == feature_columns(2)
    assert manifest["train_period"]["start"] == "2022-01-02"
    assert manifest["train_period"]["end"] == "2023-12-31"

    for h in (1, 2):
        imp = pd.read_csv(metrics_dir / f"feature_importance_h{h}.csv")
        assert list(imp.columns) == ["feature", "gain", "share"]
        assert set(imp["feature"]) == set(feature_columns(h))
    assert not (metrics_dir / "feature_importance_h3.csv").exists()

    rows = make_features(panel, 2).tail(5)  # includes NaN-y rows at the end, like live use
    pred = registry.load("lgbm", 2, models_dir).predict(rows)
    assert pred.index.equals(rows.index) and pred.notna().all().all()

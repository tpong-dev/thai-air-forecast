"""Fit the served models on all labelled data and save them for the app.

The backtest answers "how good is the model?"; this module produces the models that are actually
served. Each named model is fitted once per horizon on every row with an observed target, saved
via the registry as ``models/{name}_h{h}.joblib``, and described in ``models/manifest.json``.
For LightGBM the p50 gain importances are written to
``reports/metrics/feature_importance_h{h}.csv`` so the README can show what drives the forecast.

CLI: ``python -m thaiair.train [--names lgbm,deep] [--horizons 1,2,3]``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from thaiair import config
from thaiair.models import registry


def _labelled(features: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training: the target was observed (the Forecaster.fit contract)."""
    return features[features["y"].notna()].reset_index(drop=True)


def _write_importance(model: object, horizon: int, metrics_dir: Path) -> Path | None:
    """Save gain importances for models that expose them (LightGBM); skip the others."""
    importance = getattr(model, "feature_importance", None)
    if importance is None:
        return None
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"feature_importance_h{horizon}.csv"
    table = importance().rename_axis("feature").reset_index()
    table["share"] = table["gain"] / table["gain"].sum()
    table.to_csv(path, index=False)
    return path


def train_final(
    panel: pd.DataFrame | None = None,
    names: Sequence[str] = ("lgbm",),
    horizons: Sequence[int] = config.HORIZONS,
    models_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
) -> list[registry.ManifestEntry]:
    """Fit, save and describe every ``(name, horizon)`` model; returns the manifest entries.

    Args:
        panel: the daily panel; ``thaiair.dataset.build_panel()`` when None.
        names: registry names of the models to train ("lgbm", "deep").
        horizons: days ahead to train a model for.
        models_dir: where to save the models and manifest (default ``config.MODELS_DIR``).
        metrics_dir: where to write feature importances (default ``reports/metrics``).
    """
    from thaiair.features import make_features

    if panel is None:
        from thaiair.dataset import build_panel

        panel = build_panel()
    metrics_path = Path(metrics_dir) if metrics_dir is not None else config.REPORTS_DIR / "metrics"

    entries: list[registry.ManifestEntry] = []
    first, last = pd.Timestamp.max, pd.Timestamp.min
    for horizon in horizons:
        train = _labelled(make_features(panel, horizon))
        if train.empty:
            raise ValueError(f"no labelled rows for horizon {horizon}: is pm25 all NaN?")
        first, last = min(first, train["target_date"].min()), max(last, train["target_date"].max())
        for name in names:
            start = time.perf_counter()
            model = registry.make_model(name, horizon).fit(train)
            path = registry.save(model, name, horizon, models_dir)
            if name == "lgbm":
                _write_importance(model, horizon, metrics_path)
            print(f"{name} h{horizon}: {len(train)} rows, {time.perf_counter() - start:.1f}s -> {path}")
            entries.append(
                {
                    "name": name,
                    "horizon": horizon,
                    "file": path.name,
                    "feature_columns": list(model.feature_cols_),  # type: ignore[attr-defined]
                    "n_train_rows": len(train),
                }
            )

    registry.write_manifest(entries, (first.date().isoformat(), last.date().isoformat()), models_dir)
    return entries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the served models on all data and save them.")
    parser.add_argument("--names", default="lgbm", help="comma-separated model names (lgbm, deep)")
    parser.add_argument("--horizons", default=",".join(map(str, config.HORIZONS)), help="comma-separated horizons")
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    train_final(names=names, horizons=horizons)


if __name__ == "__main__":
    main()

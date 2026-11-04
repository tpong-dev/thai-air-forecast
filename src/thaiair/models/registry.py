"""Save, load and list fitted models, and describe them in a manifest.

Files live in ``config.MODELS_DIR`` as ``{name}_h{horizon}.joblib`` (one fitted forecaster per
horizon, as the harness and the app expect), next to ``manifest.json``, which records what the
models were trained on so a served forecast can always be traced back to its inputs.

The module also holds the two small helpers both ML models share (which feature columns to use,
which of them are categorical) and the name -> class factory, so the backtest, ``train.py`` and
the app build models the same way. Model classes are imported lazily: loading the registry (for
example in the app, to list what is available) must not pull in torch or lightgbm.
"""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np
import pandas as pd

from thaiair import config
from thaiair.models.base import Forecaster

MANIFEST_NAME = "manifest.json"
_FILE_RE = re.compile(r"^(?P<name>.+)_h(?P<horizon>\d+)\.joblib$")
_DEFAULT_CATEGORICAL = ("city_code",)
_VERSIONED_LIBS = ("numpy", "pandas", "scikit-learn", "lightgbm", "torch", "joblib")


class ManifestEntry(TypedDict):
    """One saved model as recorded in ``manifest.json``."""

    name: str
    horizon: int
    file: str
    feature_columns: list[str]
    n_train_rows: int


# --------------------------------------------------------------------------- shared model helpers


def resolve_feature_cols(horizon: int, feature_cols: Sequence[str] | None) -> list[str]:
    """Explicit columns win; otherwise ask ``thaiair.features`` (imported late, see module doc)."""
    if feature_cols is not None:
        return list(feature_cols)
    from thaiair.features import feature_columns

    return list(feature_columns(horizon))


def categorical_columns(feature_cols: Iterable[str]) -> list[str]:
    """The categorical subset of ``feature_cols``, using ``features.CATEGORICAL`` when it exists."""
    try:
        from thaiair.features import CATEGORICAL

        known = tuple(CATEGORICAL)
    except ImportError:
        known = _DEFAULT_CATEGORICAL
    return [c for c in feature_cols if c in known]


def tail_by_target_date(target_date: pd.Series, frac: float) -> np.ndarray:
    """Boolean mask of the latest ``frac`` of rows by target date (whole days, never split).

    Used for calibration and early stopping: holding out the *most recent* rows mimics how the
    model is used (always forecasting forward), unlike a random split that leaks the future.
    """
    days = np.sort(target_date.unique())
    if len(days) < 2:
        return np.zeros(len(target_date), dtype=bool)
    n_tail = max(1, int(round(len(days) * frac)))
    cutoff = days[len(days) - n_tail]
    return (target_date >= cutoff).to_numpy()


def as_known_categories(values: pd.Series, categories: list[Any]) -> pd.Categorical:
    """Categorical pinned to the training categories; unseen values become NaN.

    Unseen values are masked first because pandas is deprecating (and will reject) building a
    Categorical from values outside the given categories.
    """
    return pd.Categorical(values.where(values.isin(categories)), categories=categories)


def make_model(name: str, horizon: int, **kwargs: Any) -> Forecaster:
    """Build an unfitted model by registry name ("lgbm" or "deep") for one horizon."""
    if name == "lgbm":
        from thaiair.models.lgbm import LightGBMQuantile

        return LightGBMQuantile(horizon, **kwargs)
    if name == "deep":
        from thaiair.models.deep import DeepQuantile

        return DeepQuantile(horizon, **kwargs)
    raise ValueError(f"unknown model name {name!r}; expected 'lgbm' or 'deep'")


# --------------------------------------------------------------------------- files


def _dir(models_dir: str | Path | None) -> Path:
    return Path(models_dir) if models_dir is not None else config.MODELS_DIR


def model_path(name: str, horizon: int, models_dir: str | Path | None = None) -> Path:
    """Where ``save`` puts (and ``load`` looks for) one model."""
    return _dir(models_dir) / f"{name}_h{horizon}.joblib"


def save(model: Forecaster, name: str, horizon: int, models_dir: str | Path | None = None) -> Path:
    """Pickle a fitted model with joblib (compressed: LightGBM text trees shrink a lot)."""
    path = model_path(name, horizon, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load(name: str, horizon: int, models_dir: str | Path | None = None) -> Forecaster:
    """Load one fitted model; raises FileNotFoundError so the app can fall back to persistence."""
    path = model_path(name, horizon, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"no saved model at {path}")
    return joblib.load(path)


def available(models_dir: str | Path | None = None) -> list[tuple[str, int]]:
    """All ``(name, horizon)`` pairs that have a file, sorted; empty if the folder is missing."""
    folder = _dir(models_dir)
    if not folder.is_dir():
        return []
    found = []
    for path in folder.glob("*.joblib"):
        match = _FILE_RE.match(path.name)
        if match:
            found.append((match["name"], int(match["horizon"])))
    return sorted(found)


# --------------------------------------------------------------------------- manifest


def library_versions() -> dict[str, str | None]:
    """Installed versions read from package metadata (no heavy imports); None if not installed."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for lib in _VERSIONED_LIBS:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def write_manifest(
    entries: Sequence[ManifestEntry],
    train_period: tuple[str, str],
    models_dir: str | Path | None = None,
) -> Path:
    """Write ``manifest.json`` describing the saved models.

    ``train_period`` is the (first, last) target date of the training rows, as ISO strings.
    """
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "names": sorted({e["name"] for e in entries}),
        "horizons": sorted({e["horizon"] for e in entries}),
        "train_period": {"start": train_period[0], "end": train_period[1]},
        "feature_columns": {str(e["horizon"]): e["feature_columns"] for e in entries},
        "models": list(entries),
        "library_versions": library_versions(),
    }
    path = _dir(models_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(models_dir: str | Path | None = None) -> dict[str, Any] | None:
    """The manifest as a dict, or None when no models have been trained yet."""
    path = _dir(models_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

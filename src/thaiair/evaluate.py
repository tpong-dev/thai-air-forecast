"""Run the full backtest and write the prediction table and metric CSVs.

    python -m thaiair.evaluate                       # all models, real data -> reports/
    python -m thaiair.evaluate --models persistence,cams,lgbm
    python -m thaiair.evaluate --quick --out /tmp/x  # synthetic panel, 3-month window (smoke test)

Every metric table carries a `row_set` column saying which rows it was computed on (see
`ROW_SETS`), because "model A beats CAMS" is only meaningful on the rows where both forecast.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from thaiair import backtest, baselines, config, metrics, synthetic
from thaiair.dataset import build_panel
from thaiair.models.base import Forecaster

log = logging.getLogger("thaiair.evaluate")

ALL_MODELS = ("persistence", "seasonal", "cams", "cams_bc", "lgbm", "deep")
# name -> (module, class); imported only when asked for, so the baselines run without lightgbm/torch.
ML_MODELS: dict[str, tuple[str, str]] = {
    "lgbm": ("thaiair.models.lgbm", "LightGBMQuantile"),
    "deep": ("thaiair.models.deep", "DeepQuantile"),
}
QUICK_WINDOW = ("2024-01-01", "2024-03-31")

ROW_SETS = {
    "paired_all_models": "forecast situations (city, horizon, issue_date) where every evaluated model has y and p50",
    "own_rows": "each model on all of its own rows with y and p50 (row counts differ between models)",
    "paired_with_cams": "each model paired with CAMS alone: rows where both that model and CAMS have y and p50",
}


def _ml_factory(name: str) -> Callable[[int], Forecaster] | None:
    """Import an ML model lazily; a missing dependency skips it with a warning instead of failing."""
    module_name, class_name = ML_MODELS[name]
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        warnings.warn(f"skipping {name}: {exc}", stacklevel=2)
        return None
    return lambda horizon: cls(horizon)


def make_factories(names: list[str]) -> dict[str, Callable[[int], Forecaster]]:
    """Model name -> `horizon -> Forecaster` for every requested model that can be built."""
    unknown = sorted(set(names) - set(ALL_MODELS))
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {', '.join(ALL_MODELS)}")
    factories: dict[str, Callable[[int], Forecaster]] = {}
    for name in names:
        if name in baselines.BASELINES:
            cls = baselines.BASELINES[name]
            factories[name] = lambda _h, cls=cls: cls()
        elif (factory := _ml_factory(name)) is not None:
            factories[name] = factory
    return factories


def _tag(df: pd.DataFrame, row_set: str, **extra: str) -> pd.DataFrame:
    """Put the row-set label (and constant group columns such as season='all') up front."""
    df = df.assign(**extra) if extra else df
    df.insert(0, "row_set", row_set)
    return df


def _with_all_seasons(fn: Callable[..., pd.DataFrame], preds: pd.DataFrame, **kw: object) -> pd.DataFrame:
    """Same metric over all seasons (season='all') and per season, in one table."""
    overall = fn(preds, by=["horizon"], **kw).assign(season="all")
    out = pd.concat([overall, fn(preds, by=["horizon", "season"], **kw)], ignore_index=True)
    cols = [c for c in out.columns if c != "season"]
    cols.insert(cols.index("horizon") + 1, "season")
    return out[cols]


def metric_tables(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All CSV tables. Comparisons use `paired_all_models`; `by_horizon` also shows `own_rows`."""
    models = sorted(preds["model"].unique())
    both = metrics.paired(preds, models)
    tables = {
        "by_horizon": pd.concat(
            [
                _tag(metrics.score_table(both, by=["horizon"]), "paired_all_models"),
                _tag(metrics.score_table(preds, by=["horizon"]), "own_rows"),
            ],
            ignore_index=True,
        ),
        "by_city": _tag(metrics.score_table(both, by=["city", "horizon"]), "paired_all_models"),
        "by_season": _tag(metrics.score_table(both, by=["season", "horizon"]), "paired_all_models"),
        "alerts": _tag(_with_all_seasons(metrics.alert_scores, both), "paired_all_models"),
        "coverage": _tag(metrics.coverage(both, by=["horizon"]), "paired_all_models"),
    }
    if "cams" in models:
        tables["skill_vs_cams"] = _tag(_with_all_seasons(metrics.skill, preds, ref="cams"), "paired_with_cams")
    else:
        log.warning("cams not evaluated: skipping skill_vs_cams.csv")
    return tables


def write_outputs(preds: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Write `backtest_predictions.parquet`, `metrics/*.csv` and a short `metrics/README.md`."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "backtest_predictions.parquet", index=False)
    tables = metric_tables(preds)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=False, float_format="%.4f")
    lines = ["# Backtest metric tables", "", "Generated by `python -m thaiair.evaluate`. Column `row_set`:", ""]
    lines += [f"- `{key}`: {text}" for key, text in ROW_SETS.items()]
    lines += [
        "",
        f"Models: {', '.join(sorted(preds['model'].unique()))}. Alert threshold: {config.UNHEALTHY_PM25} ug/m3.",
    ]
    (metrics_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tables


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m thaiair.evaluate", description=__doc__.splitlines()[0])
    p.add_argument("--models", default=",".join(ALL_MODELS), help="comma-separated subset of " + ",".join(ALL_MODELS))
    p.add_argument("--quick", action="store_true", help="synthetic panel and a 3-month window (smoke test)")
    p.add_argument(
        "--out", type=Path, default=None, help="output directory (default: reports/, or a temp dir with --quick)"
    )
    p.add_argument("--processed-dir", type=Path, default=None, help="processed data directory (default: config)")
    p.add_argument("--test-start", default=None, help=f"default {config.TEST_START} ({QUICK_WINDOW[0]} with --quick)")
    p.add_argument("--test-end", default=None, help=f"default {config.TEST_END} ({QUICK_WINDOW[1]} with --quick)")
    p.add_argument("--deep-refit", default="QS", help="refit frequency of the deep model (others refit monthly)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(config.N_THREADS))

    if args.quick:
        panel = synthetic.make_panel()
        start, end = args.test_start or QUICK_WINDOW[0], args.test_end or QUICK_WINDOW[1]
        out_dir = args.out or Path(tempfile.mkdtemp(prefix="thaiair_quick_"))
    else:
        panel = build_panel(args.processed_dir)
        start, end = args.test_start or config.TEST_START, args.test_end or config.TEST_END
        out_dir = args.out or config.REPORTS_DIR

    factories = make_factories([m.strip() for m in args.models.split(",") if m.strip()])
    if not factories:
        log.error("no model could be built")
        return 1
    preds = backtest.run(panel, factories, test_start=start, test_end=end, refit={"deep": args.deep_refit})
    tables = write_outputs(preds, Path(out_dir))

    summary = tables["by_horizon"]
    with pd.option_context("display.width", 140, "display.max_rows", 100):
        print(summary[summary["row_set"] == "paired_all_models"].to_string(index=False, float_format="%.2f"))
    print(f"\nwrote {Path(out_dir) / 'backtest_predictions.parquet'} and {Path(out_dir) / 'metrics'}/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

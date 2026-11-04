"""Turn the backtest predictions into ``reports/results.md`` and three figures.

Run after the backtest::

    python -m thaiair.report [--preds reports/backtest_predictions.parquet] [--out reports] [--fill-readme]

Design choices, and why:

- **Every comparison is paired.** A model is only scored on the (city, horizon, issue date) rows where
  CAMS and every other model also have a forecast and the truth is known. Otherwise a model that skips
  hard days (for example a data outage during a smog peak) would look better than it is.
- **The CAMS caveat is printed next to the numbers**, not hidden in a doc: the archived CAMS series has a
  0-1 day lead at every horizon, so at 48/72 h the fight is tilted towards CAMS (docs/sources.md).
- **"Where CAMS wins" is generated**, so the report cannot quietly leave out the groups we lose.
- Tables are wrapped in ``<!-- RESULTS:name -->`` ... ``<!-- /RESULTS -->`` markers. The README uses the
  same markers, and ``--fill-readme`` copies the blocks over, so README numbers never drift from here.

Metric functions come from ``thaiair.metrics`` (contract section 6) when they exist and return the
expected shape. Otherwise a local version with the same definition is used and the report says so.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thaiair import config

log = logging.getLogger(__name__)

PRED_COLUMNS = ["model", "city", "horizon", "issue_date", "target_date", "y", "p10", "p50", "p90"]
KEYS = ["city", "horizon", "issue_date"]  # one forecast situation; target_date follows from these
REF = "cams"
BASELINES = ("persistence", "seasonal", "cams", "cams_bc")
MODEL_ORDER = ["persistence", "seasonal", "cams", "cams_bc", "lgbm", "deep"]
SEASONS = ("burning", "rest")
FOCUS_CITY = "chiang_mai"
EPISODE_DAYS = 60

LABELS = {
    "persistence": "Persistence",
    "seasonal": "Seasonal average",
    "cams": "CAMS",
    "cams_bc": "CAMS bias-corrected",
    "lgbm": "LightGBM",
    "deep": "Small neural net",
}
# Okabe-Ito palette (colour-blind safe). One fixed colour per model across every figure.
COLORS = {
    "persistence": "#999999",
    "seasonal": "#E69F00",
    "cams": "#0072B2",
    "cams_bc": "#56B4E9",
    "lgbm": "#D55E00",
    "deep": "#009E73",
}
EXTRA_COLORS = ["#CC79A7", "#F0E442", "#000000"]
MARKERS = {"persistence": "s", "seasonal": "D", "cams": "o", "cams_bc": "v", "lgbm": "^", "deep": "P"}
LINESTYLES = {"cams": "--", "cams_bc": "--"}

CAMS_NOTE = (
    "CAMS is the Open-Meteo archive, a stitched series of the most recent CAMS run, so its real lead time "
    "is about 0-1 day at every horizon. Our models forecast a true 24/48/72 h ahead, so at h=2 and h=3 "
    "the comparison favours CAMS (see docs/sources.md). CAMS bias-corrected is built on the same archived "
    "value, so it shares this head start."
)

MARK_RE = re.compile(r"(<!-- RESULTS:(\w+) -->)(.*?)(<!-- /RESULTS -->)", re.DOTALL)


# --------------------------------------------------------------------------------------------------
# Metrics: prefer thaiair.metrics, fall back to local versions with the contract's definitions.
# --------------------------------------------------------------------------------------------------


@dataclass
class _Metrics:
    """Calls ``thaiair.metrics`` when possible and records every local fallback for the report."""

    module: Any = None
    fallbacks: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> _Metrics:
        try:
            from thaiair import metrics  # written by the harness module; may not exist yet
        except ImportError:
            log.warning("thaiair.metrics not importable; using local metric functions")
            return cls(module=None)
        return cls(module=metrics)

    def _note(self, name: str, why: str) -> None:
        msg = f"`{name}`: {why}"
        if msg not in self.fallbacks:
            self.fallbacks.append(msg)
            log.warning("metrics.%s: %s; using local version", name, why)

    def _call(self, name: str, validate: Callable[[Any], bool], *args: Any, **kwargs: Any) -> Any:
        """Call ``metrics.<name>`` with the kwargs it accepts; ``None`` if missing, failing or wrong shape."""
        fn = getattr(self.module, name, None) if self.module is not None else None
        if fn is None:
            self._note(name, "missing" if self.module is not None else "thaiair.metrics not importable")
            return None
        try:
            params = inspect.signature(fn).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            out = fn(*args, **kwargs)
        except Exception as exc:  # any failure means: use the local definition
            self._note(name, f"raised {type(exc).__name__}: {exc}")
            return None
        if not validate(out):
            self._note(name, "returned an unexpected shape")
            return None
        return out

    # ---- public helpers used by the report -------------------------------------------------------

    def season(self, dates: pd.Series) -> pd.Series:
        """'burning' if the target month is in config.BURNING_MONTHS, else 'rest'."""
        dates = pd.to_datetime(dates)
        fn = getattr(self.module, "season", None) if self.module is not None else None
        if fn is not None:
            try:
                out = fn(dates)
                if isinstance(out, pd.Series | np.ndarray) and len(out) == len(dates):
                    return pd.Series(np.asarray(out, dtype=object), index=dates.index)
                return dates.map(fn)
            except Exception as exc:  # maybe it only takes one date at a time
                try:
                    return dates.map(fn)
                except Exception:
                    self._note("season", f"raised {type(exc).__name__}")
        else:
            self._note("season", "missing" if self.module is not None else "thaiair.metrics not importable")
        return pd.Series(
            np.where(dates.dt.month.isin(config.BURNING_MONTHS), "burning", "rest"), index=dates.index, dtype=object
        )

    def paired(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Rows where the truth, CAMS and every model are all present (see module docstring)."""

        def ok(out: Any) -> bool:
            return isinstance(out, pd.DataFrame) and set(PRED_COLUMNS) <= set(out.columns) and len(out) > 0

        out = self._call("paired", ok, preds, ref=REF)
        if out is not None:
            out = out.dropna(subset=["y", "p50"])
            if (out["model"] == REF).any():
                return out.reset_index(drop=True)
            self._note("paired", "dropped the reference model")
        return paired_local(preds)

    def score_table(self, df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
        """n, mae, rmse, bias per model and group."""
        by = list(by)
        need = {"model", *by, "n", "mae", "rmse"}

        def ok(out: Any) -> bool:
            return isinstance(out, pd.DataFrame) and need <= set(out.reset_index().columns)

        out = self._call("score_table", ok, df, by=by)
        if out is not None:
            out = out.reset_index()
            return out[[c for c in out.columns if c in need or c == "bias"]]
        return score_local(df, ["model", *by])

    def alert_scores(self, df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
        """events, alerts, precision, recall, f1 per model and group for 'y > UNHEALTHY_PM25'."""
        by = list(by)
        need = {"model", *by, "precision", "recall", "f1"}

        def ok(out: Any) -> bool:
            return isinstance(out, pd.DataFrame) and need <= set(out.reset_index().columns)

        local = alerts_local(df, ["model", *by])
        out = self._call("alert_scores", ok, df, threshold=config.UNHEALTHY_PM25, by=by)
        if out is None:
            return local
        out = out.reset_index()[[*["model", *by], "precision", "recall", "f1"]]
        counts = local[["model", *by, "events", "alerts"]]
        return counts.merge(out, on=["model", *by], how="left")

    def coverage(self, df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
        """Share of y inside [p10, p90] and the mean interval width, per model and group."""
        by = list(by)
        need = {"model", *by, "coverage", "width"}
        sub = df.dropna(subset=["p10", "p90"])
        if sub.empty:
            return pd.DataFrame(columns=["model", *by, "n", "coverage", "width"])

        def norm(out: Any) -> Any:
            # thaiair.metrics calls the width column `mean_width`
            return out.reset_index().rename(columns={"mean_width": "width"}) if isinstance(out, pd.DataFrame) else out

        def ok(out: Any) -> bool:
            return isinstance(out, pd.DataFrame) and need <= set(norm(out).columns)

        local = coverage_local(sub, ["model", *by])
        out = self._call("coverage", ok, sub, by=by)
        if out is None:
            return local
        out = norm(out)[["model", *by, "coverage", "width"]]
        return local[["model", *by, "n"]].merge(out, on=["model", *by], how="left")

    def skill(self, df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
        """``model, *by, mae, mae_ref, skill`` with skill = 1 - MAE(model) / MAE(CAMS), CAMS itself left out."""
        by = list(by)
        cols = ["model", *by, "mae", "mae_ref", "skill"]

        def norm(out: Any) -> Any:
            return out.reset_index().rename(columns={"mae_model": "mae"}) if isinstance(out, pd.DataFrame) else out

        def ok(out: Any) -> bool:
            return isinstance(out, pd.DataFrame) and set(cols) <= set(norm(out).columns)

        out = self._call("skill", ok, df, ref=REF, metric="mae", by=by)
        if out is not None:
            return norm(out)[cols]
        t = self.score_table(df, by=by)
        ref = t[t["model"] == REF][[*by, "mae"]].rename(columns={"mae": "mae_ref"})
        s = t[t["model"] != REF].merge(ref, on=by)
        s["skill"] = 1 - s["mae"] / s["mae_ref"]
        return s[cols]


def paired_local(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep the (city, horizon, issue_date) rows where y is known and every model, incl. CAMS, has a p50."""
    df = preds.dropna(subset=["y", "p50"])
    if REF not in set(df["model"]):
        raise ValueError(f"no '{REF}' predictions with a known y: the comparison needs the CAMS baseline")
    n_models = df["model"].nunique()
    counts = df.groupby(KEYS)["model"].nunique()
    full = counts[counts == n_models].index
    keep = df.set_index(KEYS).index.isin(full)
    out = df[keep].reset_index(drop=True)
    n_ref = int((df["model"] == REF).sum())
    n_kept = int((out["model"] == REF).sum())
    if n_ref and n_kept < 0.8 * n_ref:
        log.warning("pairing keeps only %d of %d CAMS rows: some model has gaps", n_kept, n_ref)
    return out


def score_local(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    err = df["p50"] - df["y"]
    g = df.assign(_e=err, _ae=err.abs(), _se=err**2).groupby(by, observed=True, sort=False)
    out = g.agg(n=("_e", "size"), mae=("_ae", "mean"), rmse=("_se", "mean"), bias=("_e", "mean")).reset_index()
    out["rmse"] = np.sqrt(out["rmse"])
    return out


def alerts_local(df: pd.DataFrame, by: list[str], threshold: float = config.UNHEALTHY_PM25) -> pd.DataFrame:
    """An alert = forecast p50 above the threshold; an event = observed y above it."""
    obs = df["y"] > threshold
    pred = df["p50"] > threshold
    g = df.assign(_obs=obs, _pred=pred, _tp=obs & pred).groupby(by, observed=True, sort=False)
    out = g.agg(events=("_obs", "sum"), alerts=("_pred", "sum"), tp=("_tp", "sum")).reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["precision"] = np.where(out["alerts"] > 0, out["tp"] / out["alerts"].replace(0, np.nan), np.nan)
        out["recall"] = np.where(out["events"] > 0, out["tp"] / out["events"].replace(0, np.nan), np.nan)
        p, r = out["precision"], out["recall"]
        out["f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), np.nan)
    return out.drop(columns="tp")


def coverage_local(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    inside = (df["y"] >= df["p10"]) & (df["y"] <= df["p90"])
    g = df.assign(_in=inside, _w=df["p90"] - df["p10"]).groupby(by, observed=True, sort=False)
    return g.agg(n=("_in", "size"), coverage=("_in", "mean"), width=("_w", "mean")).reset_index()


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def label(model: str) -> str:
    return LABELS.get(model, model)


def color(model: str, models: Sequence[str] = ()) -> str:
    if model in COLORS:
        return COLORS[model]
    extras = [m for m in models if m not in COLORS]
    return EXTRA_COLORS[extras.index(model) % len(EXTRA_COLORS)] if model in extras else EXTRA_COLORS[0]


def city_name(key: str) -> str:
    return config.CITIES[key].name if key in config.CITIES else key


def order_models(models: Sequence[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def pick_best_ml(paired: pd.DataFrame, metrics: _Metrics) -> str:
    """LightGBM if present; else the non-baseline model with the lowest paired MAE; else the best non-CAMS."""
    models = set(paired["model"])
    if "lgbm" in models:
        return "lgbm"
    scores = metrics.score_table(paired, by=["horizon"]).groupby("model", as_index=False)["mae"].mean()
    ml = scores[~scores["model"].isin(BASELINES)]
    pool = ml if not ml.empty else scores[scores["model"] != REF]
    if pool.empty:
        return REF
    return str(pool.sort_values("mae").iloc[0]["model"])


def _f(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    """Signed percent; one decimal below 1% so a narrow loss never prints as a tie ("0%" only if it is one)."""
    if x is None or pd.isna(x):
        return "n/a"
    v = x * 100
    if round(v, 1) == 0:
        return "0%"
    return f"{v:+.1f}%" if abs(v) < 1 else f"{v:+.0f}%"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """GitHub markdown table; first column left-aligned, the rest right-aligned (they are numbers)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([":---", *["---:"] * (len(header) - 1)]) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _wrap(name: str, body: str) -> str:
    return f"<!-- RESULTS:{name} -->\n{body.strip()}\n<!-- /RESULTS -->"


def _bold_min(values: dict[str, float], nd: int = 1) -> dict[str, str]:
    """Format numbers and bold the smallest (best) one."""
    finite = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
    best = min(finite.values()) if finite else None
    return {k: (f"**{_f(v, nd)}**" if best is not None and v == best else _f(v, nd)) for k, v in values.items()}


def _skill(model_mae: float, ref_mae: float) -> float:
    return float("nan") if not ref_mae or pd.isna(ref_mae) else 1.0 - model_mae / ref_mae


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------


@dataclass
class Report:
    preds: pd.DataFrame
    metrics: _Metrics
    paired: pd.DataFrame = field(init=False)
    models: list[str] = field(init=False)
    horizons: list[int] = field(init=False)
    best: str = field(init=False)

    def __post_init__(self) -> None:
        preds = prepare(self.preds)
        self.preds = preds
        paired = self.metrics.paired(preds)
        paired = paired.assign(season=self.metrics.season(paired["target_date"]).to_numpy())
        self.paired = paired
        self.models = order_models(list(paired["model"].unique()))
        self.horizons = sorted(int(h) for h in paired["horizon"].unique())
        self.best = pick_best_ml(paired, self.metrics)

    # ---- shared score tables ---------------------------------------------------------------------

    def _mae(self, by: list[str]) -> pd.DataFrame:
        return self.metrics.score_table(self.paired, by=by)

    def _lookup(self, table: pd.DataFrame, col: str, **where: Any) -> float:
        m = np.ones(len(table), dtype=bool)
        for k, v in where.items():
            m &= (table[k] == v).to_numpy()
        vals = table.loc[m, col]
        return float(vals.iloc[0]) if len(vals) else float("nan")

    # ---- sections --------------------------------------------------------------------------------

    def headline(self) -> str:
        t = self._mae(["horizon"])
        ts = self._mae(["horizon", "season"])
        b = self.best
        parts = []
        for h in (self.horizons[0], self.horizons[-1]):
            mb, mc = self._lookup(t, "mae", model=b, horizon=h), self._lookup(t, "mae", model=REF, horizon=h)
            parts.append(f"at h={h} {label(b)} MAE is {_f(mb)} µg/m³ vs {_f(mc)} for CAMS (skill {_pct(_skill(mb, mc))})")
        h = self.horizons[0]
        mb = self._lookup(ts, "mae", model=b, horizon=h, season="burning")
        mc = self._lookup(ts, "mae", model=REF, horizon=h, season="burning")
        n = int(self.paired[self.paired["model"] == REF].shape[0])
        return (
            f"On {n:,} paired forecasts, " + "; ".join(parts) + ". "
            f"In the burning season (Jan-Apr) at h={h}: {_f(mb)} vs {_f(mc)} (skill {_pct(_skill(mb, mc))})."
        )

    def by_horizon(self) -> str:
        t = self._mae(["horizon"])
        hs = self.horizons
        mae = {h: _bold_min({m: self._lookup(t, "mae", model=m, horizon=h) for m in self.models}) for h in hs}
        rmse = {h: _bold_min({m: self._lookup(t, "rmse", model=m, horizon=h) for m in self.models}) for h in hs}
        rows = [[label(m) + ("†" if m == REF else ""), *[mae[h][m] for h in hs], *[rmse[h][m] for h in hs]] for m in self.models]
        n = {h: int(self._lookup(t, "n", model=REF, horizon=h)) for h in hs}
        header = ["Model", *[f"MAE h={h}" for h in hs], *[f"RMSE h={h}" for h in hs]]
        note = (
            "µg/m³, lower is better, best in bold. Paired rows per horizon: "
            + ", ".join(f"h={h}: {n[h]:,}" for h in hs)
            + f".\n\n† {CAMS_NOTE}"
        )
        return md_table(header, rows) + "\n\n" + note

    def by_city(self) -> str:
        t = self._mae(["horizon", "city"])
        out = []
        cities = [c for c in config.CITIES if c in set(self.paired["city"])]
        cities += sorted(set(self.paired["city"]) - set(cities))
        for h in [h for h in (1, 3) if h in self.horizons] or self.horizons[:1]:
            rows = []
            for c in cities:
                vals = _bold_min({m: self._lookup(t, "mae", model=m, horizon=h, city=c) for m in self.models})
                n = self._lookup(t, "n", model=REF, horizon=h, city=c)
                rows.append([city_name(c), f"{int(n):,}" if not pd.isna(n) else "0", *[vals[m] for m in self.models]])
            out.append(f"**h={h} ({24 * h} h ahead), MAE µg/m³**\n\n" + md_table(["City", "n", *map(label, self.models)], rows))
        return "\n\n".join(out)

    def by_season(self) -> str:
        t = self._mae(["horizon", "season"])
        hs = self.horizons
        cols = [(s, h) for s in SEASONS for h in hs]
        vals = {(s, h): _bold_min({m: self._lookup(t, "mae", model=m, horizon=h, season=s) for m in self.models}) for s, h in cols}
        rows = [[label(m), *[vals[c][m] for c in cols]] for m in self.models]
        header = ["Model", *[f"{'Burning' if s == 'burning' else 'Rest'} h={h}" for s, h in cols]]
        n = {s: int(self._lookup(t, "n", model=REF, horizon=hs[0], season=s) or 0) for s in SEASONS}
        mean_y = self.paired[self.paired["model"] == REF].groupby("season")["y"].mean()
        note = (
            "MAE µg/m³, best in bold. Burning season = target day in "
            + ", ".join(pd.Timestamp(2000, m, 1).strftime("%b") for m in config.BURNING_MONTHS)
            + f". Paired rows at h={hs[0]}: burning {n['burning']:,}, rest {n['rest']:,}. "
            + f"Mean observed PM2.5: burning {_f(mean_y.get('burning'))}, rest {_f(mean_y.get('rest'))} µg/m³."
        )
        return md_table(header, rows) + "\n\n" + note

    def alerts(self) -> str:
        a = self.metrics.alert_scores(self.paired, by=["horizon"])
        rows = []
        for h in self.horizons:
            f1 = {m: self._lookup(a, "f1", model=m, horizon=h) for m in self.models}
            best = max((v for v in f1.values() if not pd.isna(v)), default=None)
            for m in self.models:
                g = dict(model=m, horizon=h)
                f = _f(f1[m], 2)
                rows.append([
                    label(m), str(h), f"{int(self._lookup(a, 'events', **g)):,}", f"{int(self._lookup(a, 'alerts', **g)):,}",
                    _f(self._lookup(a, "precision", **g), 2), _f(self._lookup(a, "recall", **g), 2),
                    f"**{f}**" if best is not None and f1[m] == best else f,
                ])
        note = (
            f"An event is an observed daily mean above {config.UNHEALTHY_PM25:g} µg/m³ (Thai AQI orange or worse). "
            "An alert is a p50 forecast above the same line. Precision = share of alerts that were real; "
            "recall = share of events that got an alert. Best F1 per horizon in bold."
        )
        return md_table(["Model", "h", "Events", "Alerts", "Precision", "Recall", "F1"], rows) + "\n\n" + note

    def coverage(self) -> str:
        c = self.metrics.coverage(self.paired, by=["horizon"])
        cs = self.metrics.coverage(self.paired, by=["horizon", "season"])
        if c.empty:
            return "No model produced p10/p90 intervals."
        rows = []
        for m in [m for m in self.models if m in set(c["model"])]:
            for h in self.horizons:
                g = dict(model=m, horizon=h)
                rows.append([
                    label(m), str(h), f"{int(self._lookup(c, 'n', **g)):,}",
                    _pct_plain(self._lookup(c, "coverage", **g)),
                    _pct_plain(self._lookup(cs, "coverage", season="burning", **g)),
                    _pct_plain(self._lookup(cs, "coverage", season="rest", **g)),
                    _f(self._lookup(c, "width", **g)),
                ])
        note = (
            "A p10-p90 interval should contain the truth 80% of the time. Below 80% = overconfident, above = too wide. "
            "Width = mean p90 - p10 in µg/m³ (narrower is better only if coverage holds)."
        )
        header = ["Model", "h", "n", "Coverage", "Burning", "Rest", "Width"]
        return md_table(header, rows) + "\n\n" + note

    def skill_table(self) -> str:
        t = self._mae(["horizon", "season"])
        cols = [(h, s) for h in self.horizons for s in SEASONS]
        rows = []
        for m in self.models:
            if m == REF:
                cells = []
                for h, s in cols:
                    cells.append(f"ref ({_f(self._lookup(t, 'mae', model=REF, horizon=h, season=s))})")
                rows.append([f"{label(REF)}†", *cells])
                continue
            cells = []
            for h, s in cols:
                sk = _skill(self._lookup(t, "mae", model=m, horizon=h, season=s), self._lookup(t, "mae", model=REF, horizon=h, season=s))
                cells.append(f"**{_pct(sk)}**" if not pd.isna(sk) and round(sk * 1000) > 0 else _pct(sk))
            rows.append([label(m), *cells])
        header = ["Model", *[f"h={h} {s}" for h, s in cols]]
        note = (
            "Skill = 1 - MAE(model) / MAE(CAMS) on the same rows. Positive (bold) = better than CAMS. "
            "The CAMS row shows its own MAE in µg/m³.\n\n"
            f"† {CAMS_NOTE} A positive skill at h=2/3 is therefore a conservative result; a negative one may be "
            "partly due to the lead-time gap."
        )
        return md_table(header, rows) + "\n\n" + note

    def skill_groups(self) -> pd.DataFrame:
        """Skill vs CAMS for every model in two groupings: horizon x season and city x horizon."""
        frames = []
        for by in (["horizon", "season"], ["city", "horizon"]):
            frames.append(self.metrics.skill(self.paired, by=by).assign(grouping="+".join(by)))
        return pd.concat(frames, ignore_index=True)

    def cams_wins(self) -> str:
        s = self.skill_groups()
        lines = []
        own = [m for m in self.models if m not in BASELINES or m == "cams_bc"]
        for m in [m for m in self.models if m != REF]:
            sm = s[s["model"] == m]
            lose = sm[sm["skill"] < 0]
            n_hs = int((sm["grouping"] == "horizon+season").sum())
            n_ch = int((sm["grouping"] == "city+horizon").sum())
            l_hs = int((lose["grouping"] == "horizon+season").sum())
            l_ch = int((lose["grouping"] == "city+horizon").sum())
            head = (
                f"- **{label(m)}**: CAMS wins in {l_hs} of {n_hs} horizon-season groups "
                f"and {l_ch} of {n_ch} city-horizon groups."
            )
            lines.append(head)
            if m in own:
                for r in lose.sort_values(["grouping", "skill"], ascending=[False, True]).itertuples():
                    where = (
                        f"h={r.horizon}, {'burning season' if r.season == 'burning' else 'rest of year'}"
                        if r.grouping == "horizon+season"
                        else f"{city_name(r.city)}, h={r.horizon}"
                    )
                    lines.append(f"  - {where}: skill {_pct(r.skill)} (MAE {_f(r.mae)} vs CAMS {_f(r.mae_ref)})")
        if not lines:
            lines.append("- No model besides CAMS is in the predictions.")
        lines.append("")
        lines.append(f"Read this with the CAMS caveat: {CAMS_NOTE}")
        return "\n".join(lines)

    def summary(self) -> str:
        p = self.paired
        cities = ", ".join(city_name(c) for c in [c for c in config.CITIES if c in set(p["city"])] + sorted(set(p["city"]) - set(config.CITIES)))
        return (
            f"- Issue dates: {p['issue_date'].min():%Y-%m-%d} to {p['issue_date'].max():%Y-%m-%d} "
            "(rolling origin, refit every month, train only on days before the month starts)\n"
            f"- Cities: {cities}\n"
            f"- Models: {', '.join(f'{label(m)} (`{m}`)' for m in self.models)}\n"
            f"- Best own model (used in the figures): {label(self.best)}\n"
            f"- Rows in the predictions file: {len(self.preds):,}; paired rows used here: {len(p):,}"
        )

    def markdown(self, figures: dict[str, Path], out_dir: Path) -> str:
        sections = [
            ("Headline", "headline", self.headline()),
            ("1. Error by horizon", "by_horizon", self.by_horizon()),
            ("2. Error by city", "by_city", self.by_city()),
            ("3. Burning season vs the rest of the year", "by_season", self.by_season()),
            ("4. Unhealthy-day alerts", "alerts", self.alerts()),
            ("5. Prediction intervals (p10-p90)", "coverage", self.coverage()),
            ("6. Skill vs CAMS by horizon and season", "skill", self.skill_table()),
            ("7. Where CAMS wins", "cams_wins", self.cams_wins()),
        ]
        parts = [
            "# Backtest results",
            "",
            "Generated by `python -m thaiair.report` from the backtest predictions. Do not edit by hand.",
            "Every number is on held-out months, and every model is scored on the same rows (see below).",
            "",
            self.summary(),
        ]
        for title, key, body in sections:
            parts += ["", f"## {title}", "", _wrap(key, body)]
        parts += ["", "## Figures", ""]
        for name, path in figures.items():
            rel = path.relative_to(out_dir).as_posix() if path.is_relative_to(out_dir) else path.as_posix()
            parts.append(f"![{name}]({rel})")
            parts.append("")
        if self.metrics.fallbacks:
            parts += ["## Notes on computation", ""]
            parts.append("These metrics were computed by `thaiair.report` itself, with the same definitions as the contract:")
            parts += [f"- {f}" for f in self.metrics.fallbacks]
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"


def _pct_plain(x: float) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def prepare(preds: pd.DataFrame) -> pd.DataFrame:
    missing = set(PRED_COLUMNS) - set(preds.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    df = preds[PRED_COLUMNS].copy()
    df["model"] = df["model"].astype(str)
    df["city"] = df["city"].astype(str)
    df["horizon"] = df["horizon"].astype(int)
    for c in ("issue_date", "target_date"):
        df[c] = pd.to_datetime(df[c])
    for c in ("y", "p10", "p50", "p90"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
        "legend.frameon": False,
    })
    return plt


def _footnote(fig: Any, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#555555", ha="left", va="bottom", wrap=True)


def episode_window(actual: pd.Series, days: int = EPISODE_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The ``days``-long window with the highest mean observed PM2.5 (whole series if it is shorter)."""
    s = actual.sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    if len(s) <= days:
        return s.index.min(), s.index.max()
    roll = s.rolling(days, min_periods=max(1, int(days * 0.75))).mean()
    end = roll.idxmax()
    return end - pd.Timedelta(days=days - 1), end


def fig_smog_episode(rep: Report, path: Path) -> Path:
    """Observed vs the best own model (with its interval) vs CAMS over the worst stretch of smog."""
    plt = _plt()
    import matplotlib.dates as mdates

    p = rep.preds.dropna(subset=["y"])
    city = FOCUS_CITY if FOCUS_CITY in set(p["city"]) else str(p.groupby("city")["y"].mean().idxmax())
    pc = p[p["city"] == city]
    actual = pc.drop_duplicates("target_date").set_index("target_date")["y"]
    start, end = episode_window(actual)
    hs = [h for h in (1, 3) if h in rep.horizons] or rep.horizons[:1]
    b = rep.best
    fig, axes = plt.subplots(1, len(hs), figsize=(6.2 * len(hs), 4.4), sharey=True, squeeze=False)
    scores = []
    for ax, h in zip(axes[0], hs):
        w = pc[(pc["horizon"] == h) & pc["target_date"].between(start, end)]
        mdl = w[w["model"] == b].sort_values("target_date")
        cams = w[w["model"] == REF].sort_values("target_date")
        obs = actual[(actual.index >= start) & (actual.index <= end)].sort_index()
        ax.axhline(config.UNHEALTHY_PM25, color="#444444", lw=0.8, ls=":", label=f"Unhealthy line ({config.UNHEALTHY_PM25:g})")
        has_band = mdl["p10"].notna() & mdl["p90"].notna()
        if has_band.any():
            ax.fill_between(mdl["target_date"], mdl["p10"], mdl["p90"], where=has_band.to_numpy(), color=color(b),
                            alpha=0.22, lw=0, label=f"{label(b)} p10-p90 interval")
        ax.plot(obs.index, obs.to_numpy(), color="black", lw=1.6, label="Observed")
        ax.plot(mdl["target_date"], mdl["p50"], color=color(b), lw=1.5, label=f"{label(b)} (p50)")
        ax.plot(cams["target_date"], cams["p50"], color=color(REF), lw=1.3, ls="--", label="CAMS")
        both = mdl.merge(cams, on="target_date", suffixes=("_m", "_c")).dropna(subset=["p50_m", "p50_c", "y_m"])
        mae_m = (both["p50_m"] - both["y_m"]).abs().mean()
        mae_c = (both["p50_c"] - both["y_m"]).abs().mean()
        scores.append((h, mae_m, mae_c))
        ax.set_title(f"{h} day{'s' if h > 1 else ''} ahead ({24 * h} h): MAE {label(b)} {_f(mae_m)}, CAMS {_f(mae_c)}")
        ax.set_xlabel("Target date")
        loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axes[0][0].set_ylabel("Daily mean PM2.5 (µg/m³)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5, bbox_to_anchor=(0.5, 0.035))
    h0, m0, c0 = scores[0]
    verdict = "beats" if m0 < c0 else "does not beat"
    fig.suptitle(
        f"{city_name(city)}, worst {EPISODE_DAYS} days of the test period: "
        f"{label(b)} {verdict} CAMS at {24 * h0} h ({_f(m0)} vs {_f(c0)} µg/m³ MAE)\n"
        f"{start:%d %b %Y} to {end:%d %b %Y}",
        fontsize=11.5,
    )
    _footnote(fig, "CAMS = archived Open-Meteo series with a ~0-1 day lead at every horizon (see docs/sources.md).")
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_error_by_horizon(rep: Report, path: Path) -> Path:
    """MAE vs horizon for every model; burning season and the rest side by side (same paired rows)."""
    plt = _plt()
    t = rep._mae(["horizon", "season"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, s in zip(axes, SEASONS):
        for m in rep.models:
            ys = [rep._lookup(t, "mae", model=m, horizon=h, season=s) for h in rep.horizons]
            ax.plot(rep.horizons, ys, color=color(m, rep.models), marker=MARKERS.get(m, "o"), ls=LINESTYLES.get(m, "-"),
                    lw=2.2 if m == rep.best else 1.4, label=label(m))
        ax.set_xticks(rep.horizons, [f"{h}\n({24 * h} h)" for h in rep.horizons])
        ax.set_xlabel("Forecast horizon (days ahead)")
        ax.set_title("Burning season (Jan-Apr)" if s == "burning" else "Rest of the year (May-Dec)")
    axes[0].set_ylabel("MAE (µg/m³)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5, bbox_to_anchor=(0.5, 0.045))
    hl = rep.horizons[-1]
    sb = t[(t["season"] == "burning") & (t["horizon"] == hl)]
    best_m = str(sb.sort_values("mae").iloc[0]["model"]) if len(sb) else rep.best
    mb = rep._lookup(t, "mae", model=rep.best, horizon=hl, season="burning")
    mr = rep._lookup(t, "mae", model=rep.best, horizon=hl, season="rest")
    ratio = mb / mr if mr else float("nan")
    first = [rep._lookup(t, "mae", model=rep.best, horizon=rep.horizons[0], season=s) for s in SEASONS]
    last = [rep._lookup(t, "mae", model=rep.best, horizon=hl, season=s) for s in SEASONS]
    trend = "grow with the horizon" if all(b > a for a, b in zip(first, last)) else "do not always grow with the horizon"
    size = f"{_f(ratio)}x larger" if ratio >= 1 else f"{_f(1 / ratio)}x smaller" if ratio > 0 else "n/a"
    fig.suptitle(
        f"{label(rep.best)} errors {trend} and are {size} in the burning season\n"
        f"Lowest burning-season MAE at {24 * hl} h: {label(best_m)}",
        fontsize=11.5,
    )
    _footnote(fig, "Same paired rows for every model. CAMS has a ~0-1 day lead at every horizon, so its line is flatter than a real 72 h forecast.")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)
    return path


def _mix(hex_color: str, frac: float) -> tuple[float, float, float]:
    """Blend a colour towards white: frac=1 keeps it, frac=0 is white. Used to shade horizons of one model."""
    rgb = np.array([int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return tuple(1 - frac * (1 - rgb))  # type: ignore[return-value]


def fig_skill_by_city(rep: Report, path: Path) -> Path:
    """Skill vs CAMS of the best own model per city and horizon (h=1 on top of each group)."""
    plt = _plt()
    s = rep.skill_groups()
    s = s[(s["grouping"] == "city+horizon") & (s["model"] == rep.best)]
    cities = list(s[s["horizon"] == rep.horizons[0]].sort_values("skill")["city"])
    cities += [c for c in s["city"].unique() if c not in cities]
    n_h = len(rep.horizons)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * max(len(cities), 3) * max(n_h, 2) / 2 + 1.6))
    height = 0.8 / max(n_h, 1)
    hatches = ["", "//", ".."]
    y = np.arange(len(cities))
    for i, h in enumerate(rep.horizons):
        vals = [rep._lookup(s, "skill", city=c, horizon=h) * 100 for c in cities]
        ax.barh(y - (i - (n_h - 1) / 2) * height, vals, height=height * 0.95,
                color=_mix(color(rep.best, rep.models), 1.0 - 0.3 * i), edgecolor=color(rep.best, rep.models),
                hatch=hatches[i % len(hatches)], lw=0.6, label=f"h={h} ({24 * h} h)")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y, [city_name(c) for c in cities])
    ax.set_xlabel("Skill vs CAMS (%) = 1 - MAE(model) / MAE(CAMS); right of 0 = better than CAMS")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="best", fontsize=8)
    wins = int((s["skill"] > 0).sum())
    ax.set_title(f"{label(rep.best)} beats CAMS in {wins} of {len(s)} city-horizon pairs", fontsize=11.5)
    _footnote(fig, "CAMS has a ~0-1 day lead at every horizon (stitched archive), so h=2 and h=3 favour CAMS.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------------


def build_report(preds: pd.DataFrame, out_dir: Path | str = config.REPORTS_DIR) -> dict[str, Path]:
    """Write ``results.md`` and the figures into ``out_dir``; return their paths by name."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep = Report(preds, _Metrics.load())
    figures = {
        "smog_episode": fig_smog_episode(rep, fig_dir / "smog_episode.png"),
        "error_by_horizon": fig_error_by_horizon(rep, fig_dir / "error_by_horizon.png"),
        "skill_by_city": fig_skill_by_city(rep, fig_dir / "skill_by_city.png"),
    }
    results = out_dir / "results.md"
    results.write_text(rep.markdown(figures, out_dir), encoding="utf-8")
    log.info("wrote %s and %d figures", results, len(figures))
    return {"results": results, **figures}


def result_blocks(markdown: str) -> dict[str, str]:
    """``{name: inner text}`` for every ``<!-- RESULTS:name -->`` block."""
    return {m.group(2): m.group(3) for m in MARK_RE.finditer(markdown)}


def fill_readme(readme: Path | str, results: Path | str) -> list[str]:
    """Copy each RESULTS block from results.md into the README block with the same name.

    Only text between markers changes, so hand-written prose (takeaways) is never touched.
    Returns the names that were filled.
    """
    readme, results = Path(readme), Path(results)
    blocks = result_blocks(results.read_text(encoding="utf-8"))
    filled: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name = m.group(2)
        if name not in blocks:
            return m.group(0)
        filled.append(name)
        # Figure links in results.md are relative to reports/; the README sits one level up.
        body = blocks[name].replace("](figures/", "](reports/figures/")
        return f"{m.group(1)}{body}{m.group(4)}"

    text = readme.read_text(encoding="utf-8")
    readme.write_text(MARK_RE.sub(sub, text), encoding="utf-8")
    return filled


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, default=config.REPORTS_DIR / "backtest_predictions.parquet")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--fill-readme", action="store_true", help="copy the RESULTS blocks into README.md")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = build_report(pd.read_parquet(args.preds), args.out)
    for name, p in paths.items():
        print(f"{name}: {p}")
    if args.fill_readme:
        names = fill_readme(config.ROOT / "README.md", paths["results"])
        print(f"README blocks filled: {', '.join(names) or 'none'}")


if __name__ == "__main__":
    main()

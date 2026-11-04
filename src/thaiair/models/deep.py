"""Small PyTorch quantile network: the "does deep learning help here?" comparison model.

Design, kept deliberately small (a few thousand rows per city do not support a big net):

- Inputs: numeric features standardised with training statistics. NaN is replaced by the training
  median and, for every column that had gaps in training, a 0/1 "was missing" column is added so
  the net can learn that an outage is not the same as a median day.
- Each categorical column (the city) gets a 4-dim learned embedding; index 0 is reserved for
  categories not seen in training.
- Body: two hidden layers of 64 with ReLU and dropout 0.1.
- Output: three quantiles of the standardised *change* log1p(y) - log1p(today's PM2.5). Predicting
  the change from persistence instead of the level cut backtest MAE at h=1 from 6.7 to 3.6 ug/m3
  (the level version under-forecast smog by ~4 ug/m3). The net predicts p50 plus two
  non-negative gaps (softplus), so p10 <= p50 <= p90 holds by construction, not by sorting.
- Loss: pinball (quantile) loss averaged over the three quantiles.
- Training: Adam, batch 256, at most ``max_epochs`` epochs, early stopping on a seeded random 10%
  of training *days* (all before the fold, so no leakage), keeping the weights of the best
  validation epoch. Holding out the latest days instead put the validation set in one season
  (often the clean rainy months) and biased the net low. ``refit_full=True`` would
  instead retrain on all rows for the best epoch count; it is off by default because, on synthetic
  data, that unselected end-of-training snapshot was consistently worse (p50 MAE ~15-20% higher)
  than the early-stopped net: epoch-to-epoch noise outweighs the extra 10% of rows.

Pickling: torch modules are not stored. The fitted weights are kept as numpy arrays next to the
scaler arrays and the net is rebuilt on demand, so a saved file loads without pickling torch
internals and is independent of the torch build.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from thaiair import config
from thaiair.models.base import PRED_COLS
from thaiair.models.registry import as_known_categories, categorical_columns, resolve_feature_cols

ANCHOR_COL = "pm25_lag0"  # today's PM2.5 (forward-filled <= 2 days), see features.py


def random_days(target_date: pd.Series, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of a seeded random ``frac`` of whole target days, for early stopping.

    Every training row is already before the fold, so this cannot leak the test period. A random
    set of days covers all seasons; holding out only the latest days (often the clean rainy
    season) made the net stop early on low values and under-forecast smog.
    """
    days = np.sort(target_date.unique())
    if len(days) < 10:
        return np.zeros(len(target_date), dtype=bool)
    rng = np.random.default_rng(seed)
    held = rng.choice(days, size=max(1, int(round(len(days) * frac))), replace=False)
    return target_date.isin(held).to_numpy()

EMBED_DIM = 4


class _QuantileNet(nn.Module):
    """MLP over [numeric inputs, embeddings] -> (p10, p50, p90), monotone by construction."""

    def __init__(self, n_numeric: int, cardinalities: list[int], hidden: int, dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(card, EMBED_DIM) for card in cardinalities)
        width = n_numeric + EMBED_DIM * len(cardinalities)
        self.body = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        raw = self.body(torch.cat(parts, dim=1))
        mid = raw[:, 1]
        lo = mid - nn.functional.softplus(raw[:, 0])
        hi = mid + nn.functional.softplus(raw[:, 2])
        return torch.stack([lo, mid, hi], dim=1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Mean quantile loss; its minimiser for column j is the ``quantiles[j]`` quantile of y."""
    err = y.unsqueeze(1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1) * err).mean()


class DeepQuantile:
    """p10 / p50 / p90 of PM2.5 for one horizon from a small seeded MLP.

    Args:
        horizon: days ahead (1-3); only used to pick the default feature columns.
        feature_cols: columns of the ``make_features`` table; default
            ``thaiair.features.feature_columns(horizon)``.
        seed: seeds weights, dropout and batch order, so two fits on the same data match exactly.
        hidden, dropout, lr, batch_size, max_epochs, patience: network / optimiser settings.
        val_frac: share of the latest target dates used for early stopping.
        refit_full: retrain on all rows for the best epoch count after early stopping.
    """

    name = "deep"

    def __init__(
        self,
        horizon: int,
        feature_cols: Sequence[str] | None = None,
        seed: int = 0,
        hidden: int = 64,
        dropout: float = 0.1,
        lr: float = 2e-3,
        batch_size: int = 256,
        max_epochs: int = 60,
        patience: int = 8,
        val_frac: float = 0.1,
        refit_full: bool = False,
    ) -> None:
        self.horizon = horizon
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.seed = seed
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.refit_full = refit_full
        self.quantiles = config.QUANTILES
        # Fitted state: plain numpy / python objects only (see module docstring on pickling).
        self.feature_cols_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categories_: dict[str, list[Any]] = {}
        self.medians_ = np.empty(0)
        self.missing_cols_: list[str] = []
        self.mean_ = np.empty(0)
        self.scale_ = np.empty(0)
        self.y_mean_ = 0.0
        self.y_scale_ = 1.0
        self.best_epochs_ = 0
        self.state_: dict[str, np.ndarray] | None = None
        self._net: _QuantileNet | None = None

    # ----------------------------------------------------------------------- public API

    def fit(self, train: pd.DataFrame) -> DeepQuantile:
        """Learn the scalers, early-stop on the latest rows, then (optionally) refit on all rows."""
        torch.set_num_threads(config.N_THREADS)
        self.feature_cols_ = resolve_feature_cols(self.horizon, self.feature_cols)
        cat_cols = categorical_columns(self.feature_cols_)
        self.numeric_cols_ = [c for c in self.feature_cols_ if c not in cat_cols]
        self.categories_ = {c: sorted(train[c].dropna().unique().tolist()) for c in cat_cols}
        self._fit_scaler(train)

        x_num, x_cat = self._inputs(train)
        y_log = np.log1p(np.clip(train["y"].to_numpy(dtype=float), 0, None))
        self.anchor_fill_ = float(np.nanmean(y_log))
        # Residual target: change from today's level (log space). The net starts from persistence
        # and only has to learn the correction, which a small MLP does far better than the level.
        y_res = y_log - self._anchor(train)
        self.y_mean_, self.y_scale_ = float(y_res.mean()), float(y_res.std() or 1.0)
        y = (y_res - self.y_mean_) / self.y_scale_

        val = random_days(train["target_date"], self.val_frac, self.seed)
        if val.any() and (~val).any():
            net, self.best_epochs_ = self._train(x_num[~val], x_cat[~val], y[~val], (x_num[val], x_cat[val], y[val]))
            if self.refit_full:
                net, _ = self._train(x_num, x_cat, y, None, epochs=self.best_epochs_)
        else:  # too little history to hold any out: train for the full budget
            net, self.best_epochs_ = self._train(x_num, x_cat, y, None, epochs=self.max_epochs)

        self.state_ = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        self._net = net.eval()
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """p10 / p50 / p90 in µg/m³, same index as ``test``."""
        if self.state_ is None:
            raise RuntimeError("DeepQuantile.predict called before fit")
        torch.set_num_threads(config.N_THREADS)
        x_num, x_cat = self._inputs(test)
        with torch.no_grad():
            out = self._network()(torch.from_numpy(x_num), torch.from_numpy(x_cat)).numpy().astype(float)
        out = np.expm1(out * self.y_scale_ + self.y_mean_ + self._anchor(test)[:, None])
        return pd.DataFrame(np.clip(out, 0, None), index=test.index, columns=PRED_COLS)

    def _anchor(self, df: pd.DataFrame) -> np.ndarray:
        """log1p of today's PM2.5 (the persistence forecast); training mean where it is unknown."""
        if ANCHOR_COL not in df:
            return np.full(len(df), self.anchor_fill_)
        a = np.log1p(np.clip(df[ANCHOR_COL].to_numpy(dtype=float), 0, None))
        return np.where(np.isnan(a), self.anchor_fill_, a)

    # ----------------------------------------------------------------------- pickling

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_net"] = None  # rebuilt from state_ (numpy) on first predict
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ----------------------------------------------------------------------- preprocessing

    def _fit_scaler(self, train: pd.DataFrame) -> None:
        num = train[self.numeric_cols_].apply(pd.to_numeric, errors="coerce").astype(float)
        self.medians_ = num.median().fillna(0.0).to_numpy()
        self.missing_cols_ = [c for c in self.numeric_cols_ if num[c].isna().any()]
        filled = self._numeric_block(train)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = np.where(filled.std(axis=0) > 1e-9, filled.std(axis=0), 1.0)

    def _numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        """Median-filled numeric features followed by the missing-indicator columns (unscaled)."""
        num = df[self.numeric_cols_].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
        missing = np.isnan(num)
        filled = np.where(missing, self.medians_, num)
        idx = [self.numeric_cols_.index(c) for c in self.missing_cols_]
        return np.hstack([filled, missing[:, idx].astype(float)])

    def _inputs(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(standardised float32 numeric block, int64 category indices with 0 = unseen)."""
        x_num = ((self._numeric_block(df) - self.mean_) / self.scale_).astype(np.float32)
        cat_parts = []
        for col, cats in self.categories_.items():
            codes = as_known_categories(df[col], cats).codes.astype(np.int64)
            cat_parts.append(codes + 1)  # -1 (unseen / NaN) -> 0
        x_cat = np.column_stack(cat_parts) if cat_parts else np.zeros((len(df), 0), dtype=np.int64)
        return x_num, x_cat

    # ----------------------------------------------------------------------- training

    def _new_net(self) -> _QuantileNet:
        n_numeric = len(self.numeric_cols_) + len(self.missing_cols_)
        cards = [len(c) + 1 for c in self.categories_.values()]
        return _QuantileNet(n_numeric, cards, self.hidden, self.dropout)

    def _network(self) -> _QuantileNet:
        if self._net is None:
            assert self.state_ is not None
            net = self._new_net()
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.state_.items()})
            self._net = net.eval()
        return self._net

    def _train(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        val: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        epochs: int | None = None,
    ) -> tuple[_QuantileNet, int]:
        """Train a fresh seeded net; with ``val`` keep the best epoch, else run ``epochs`` epochs.

        Returns the net and the number of epochs that produced it.
        """
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        net = self._new_net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        q = torch.tensor(self.quantiles, dtype=torch.float32)
        xn, xc, yt = torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(y.astype(np.float32))
        n_epochs = epochs if epochs is not None else self.max_epochs

        best_loss, best_epoch, best_state, stale = float("inf"), n_epochs, None, 0
        for epoch in range(1, n_epochs + 1):
            net.train()
            for batch in torch.randperm(len(yt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                pinball_loss(net(xn[batch], xc[batch]), yt[batch], q).backward()
                opt.step()
            if val is None:
                continue
            loss = self._val_loss(net, val, q)
            if loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = loss, epoch, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net.eval(), best_epoch

    @staticmethod
    def _val_loss(net: _QuantileNet, val: tuple[np.ndarray, np.ndarray, np.ndarray], q: torch.Tensor) -> float:
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(val[0]), torch.from_numpy(val[1]))
            return float(pinball_loss(pred, torch.from_numpy(val[2].astype(np.float32)), q))

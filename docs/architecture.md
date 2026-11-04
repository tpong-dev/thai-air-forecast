# Architecture and shared interfaces

This is the contract between the pieces of the project. Each module owns its files; they talk only
through the files and functions listed here.

## Forecast setup

- **Target:** daily mean PM2.5 (µg/m³, local calendar day, Asia/Bangkok) per city.
- **Issue time:** midnight at the end of day `t`. Everything observed on or before day `t` is known.
- **Horizons:** `h = 1, 2, 3` days → target day `t + h` (24 / 48 / 72 h ahead).
- **Why daily, not hourly:** the public Air4Thai archive only keeps ~2 months of hourly data; the
  multi-year archive is daily. Thai AQI for PM2.5 is also defined on the 24 h mean.

## Data flow

```
pm25.py   ──► data/processed/pm25_daily.parquet      (city, date, pm25, n_stations)
weather.py──► data/processed/weather_past.parquet    (city, date, w_*)   ERA5 reanalysis, observed day
          ──► data/processed/weather_fc.parquet      (city, date, f1_*, f2_*, f3_*)  archived JMA forecasts
cams.py   ──► data/processed/cams_daily.parquet      (city, date, cams_pm25)
                    │
dataset.build_panel()  ──► one row per (city, date)  = the "panel"
                    │
features.make_features(panel, h)  ──► one row per (city, issue_date)  = supervised table
                    │
backtest.run(...)   ──► reports/backtest_predictions.parquet  (long format, every model)
metrics.py          ──► reports/metrics/*.csv  ──► README tables, figures
train.py            ──► models/*.joblib (final models used by the app)
live.py             ──► today's panel rows from live APIs ──► app/api.py, app/dashboard.py
```

## 1. Processed files (columns are exact)

`pm25_daily.parquet`
| column | type | meaning |
|---|---|---|
| city | str | key from `config.CITIES` |
| date | datetime64[ns] (naive, local date at 00:00) | calendar day in Asia/Bangkok |
| pm25 | float | city daily mean = median over the city's stations that reported (NaN if none) |
| n_stations | int | number of stations that contributed |

`weather_past.parquet`: observed (ERA5 reanalysis) daily weather for that `date`. Only usable as a
**lagged** feature (it is not known in advance). Columns: `city, date, w_temp, w_rh, w_wind, w_u, w_v,
w_precip, w_blh, w_blh_min` (°C, %, m/s, m/s, m/s, mm/day, m daily mean, m daily min).

`weather_fc.parquet`: weather **forecast** valid for `date`, from a model run issued at least `h` days
before (Open-Meteo previous-runs API, JMA GSM, `*_previous_day{h}`). Safe to use for target day `t+h`.
Columns: `city, date` and for each `h in 1..3`: `f{h}_temp, f{h}_rh, f{h}_wind, f{h}_u, f{h}_v, f{h}_precip`.

`cams_daily.parquet`: `city, date, cams_pm25` = daily mean of the CAMS global forecast for that date
as archived by Open-Meteo (a stitched series of the most recent run, so effectively a 0-1 day lead).
Used **only** as the CAMS baseline at the target date, and as a lagged feature (`cams_pm25` on day `t`
and before). Never as a target-day feature for the models.

All files: one row per (city, date), no duplicates, sorted by city then date. Missing = NaN, not absent.

## 2. Panel (`thaiair.dataset.build_panel(processed_dir=None) -> DataFrame`)

Outer-joins the four files on `(city, date)` over a complete daily calendar per city from
`config.DATA_START` to the last date with any data. Columns = union of the above.
`thaiair.synthetic.make_panel()` returns a frame with exactly the same columns for tests.

## 3. Supervised table (`thaiair.features.make_features(panel, horizon) -> DataFrame`)

One row per `(city, issue_date)`. Identity columns:

| column | meaning |
|---|---|
| city, issue_date, target_date, horizon | `target_date = issue_date + horizon days` |
| y | observed pm25 on target_date (NaN allowed; rows with NaN y are dropped from scoring) |
| cams_target | CAMS daily mean on target_date (baseline only, **not** a model feature) |
| clim_doy | day of year of target_date |

Feature columns are listed by `thaiair.features.feature_columns(horizon)`. Rule: a feature may use
pm25 / w_* / cams_pm25 only from dates `<= issue_date`, and f{h}_* only for `target_date`.
`tests/test_features.py` enforces it by changing all data after `issue_date` and checking features
do not move.

## 4. Forecaster protocol (`thaiair.models.base`)

```python
class Forecaster(Protocol):
    name: str
    def fit(self, train: pd.DataFrame) -> "Forecaster": ...       # rows of make_features, y not NaN
    def predict(self, test: pd.DataFrame) -> pd.DataFrame: ...     # same index as test, columns p10, p50, p90
```

- `p50` is the point forecast. Models without intervals return NaN in p10/p90.
- A forecaster instance handles one horizon (the harness creates one per horizon via a factory).
- Baselines (`thaiair.baselines`): `Persistence`, `SeasonalAverage`, `Cams`, `CamsBiasCorrected`.
- Models (`thaiair.models`): `LightGBMQuantile` (lgbm.py), `DeepQuantile` (deep.py, small PyTorch net).

## 5. Backtest (`thaiair.backtest.run`)

```python
run(panel, factories: dict[str, Callable[[int], Forecaster]], horizons=config.HORIZONS,
    test_start=config.TEST_START, test_end=config.TEST_END, refit="MS") -> pd.DataFrame
```

Rolling origin, expanding window: for each month `M` in the test window, train on rows with
`target_date < M` (so every training label was observed before the fold starts), predict rows with
`issue_date` in `M`. Output (long): `model, city, horizon, issue_date, target_date, y, p10, p50, p90`.

## 6. Metrics (`thaiair.metrics`)

- `score_table(preds, by=[...]) -> DataFrame` with `n, mae, rmse, bias` per model and group.
- `skill(preds, ref="cams", metric="mae")` → `1 - metric(model) / metric(ref)` on the same rows.
- `alert_scores(preds, threshold=config.UNHEALTHY_PM25)` → precision, recall, F1 for "y > 37.5".
- `coverage(preds)` → share of y inside [p10, p90] and mean interval width.
- `season(date)` → `"burning"` if month in `config.BURNING_MONTHS` else `"rest"`.

## 7. Saved models and serving

`thaiair.train.train_final(panel)` fits each served model per horizon on all data and writes
`models/{name}_h{h}.joblib` (plus `models/manifest.json`: names, horizons, feature columns, train
period). `thaiair.models.registry.load(name, horizon)` loads one. The app falls back to
persistence if files are missing so it always starts.

`thaiair.live.build_live_panel(city) -> DataFrame` returns panel rows for the last ~60 days plus the
next 3 days using live sources (Air4Thai current/history, Open-Meteo forecast with `past_days`, JMA
forecast, CAMS forecast), with the same columns as the panel. `thaiair.live.forecast(city)` runs it
through `make_features` and the saved models.

## 8. Live-data functions (used by `thaiair.live`)

| function | returns |
|---|---|
| `thaiair.pm25.recent_daily(city_key, days=45)` | `city, date, pm25, n_stations` for complete past days only (from the hourly Air4Thai service) |
| `thaiair.pm25.latest_stations()` | `station, name, area, lat, lon, pm25, time` (latest reading per station, for the map) |
| `thaiair.weather.live_daily(city_key, past_days=30)` | `city, date, w_*, f1_*..f3_*` for dates today-past_days .. today+3. `w_*` filled for past days; `f{h}_*` filled for today .. today+3 from the latest JMA run |
| `thaiair.cams.live_daily(city_key, past_days=30)` | `city, date, cams_pm25` for today-past_days .. today+3 (CAMS forecast for future days) |
| `thaiair.models.registry.load(name, horizon)` | a fitted `Forecaster` |
| `thaiair.models.registry.available()` | list of `(name, horizon)` found in `models/` |

Live issue date = yesterday (last complete day, Asia/Bangkok). Targets = today, tomorrow, the day after.

## 9. Shared helpers

- `thaiair.http.get / get_json / download`: retries, backoff and on-disk caching. Use it for all HTTP.
- `thaiair.synthetic.make_panel()`: a fake panel with the real columns, for tests.

## 10. File ownership

| Area | Files |
|---|---|
| PM2.5 | `src/thaiair/pm25.py`, `tests/test_pm25.py`, `docs/cleaning.md` |
| Weather + CAMS | `src/thaiair/weather.py`, `src/thaiair/cams.py`, `tests/test_weather.py`, `tests/test_cams.py`, `docs/weather_cams.md` |
| Harness | `src/thaiair/dataset.py`, `features.py`, `baselines.py`, `backtest.py`, `metrics.py`, `evaluate.py`, `tests/test_dataset.py`, `test_features.py`, `test_baselines.py`, `test_backtest.py`, `test_metrics.py` |
| Models | `src/thaiair/models/lgbm.py`, `deep.py`, `registry.py`, `src/thaiair/train.py`, `tests/test_models.py`, `tests/test_train.py` |
| App | `src/thaiair/live.py`, `app/api.py`, `app/dashboard.py`, `Dockerfile`, `.dockerignore`, `Makefile`, `.github/workflows/*`, `tests/test_api.py`, `tests/test_live.py` |
| Docs | `README.md`, `LEARN.md`, `src/thaiair/report.py`, `tests/test_report.py`, `reports/results.md`, `reports/figures/*` |
| Lead (shared, read-only for others) | `pyproject.toml`, `src/thaiair/config.py`, `synthetic.py`, `http.py`, `models/base.py`, `docs/architecture.md`, `docs/sources.md` |

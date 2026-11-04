# Weather and CAMS data

This page explains where the weather and CAMS numbers come from, how hourly data becomes one value
per day, and why the weather features cannot leak the future. Code: `src/thaiair/weather.py` and
`src/thaiair/cams.py`. Build the files with:

```bash
python -m thaiair.weather          # weather_past.parquet + weather_fc.parquet
python -m thaiair.weather --only fc
python -m thaiair.cams             # cams_daily.parquet
```

Raw API answers are cached as JSON under `data/raw/openmeteo/{era5,jma_previous_runs,cams}/`, one
file per city and date chunk. A second run makes no new requests, except for the last chunk, whose
end date moves forward every day.

## Sources

| Product | Endpoint | Model | Used for |
|---|---|---|---|
| Observed weather | `archive-api.open-meteo.com/v1/archive` | `models=era5` (ECMWF ERA5 reanalysis, 0.25°) | `w_*` (lagged features only) |
| Issued weather forecasts | `previous-runs-api.open-meteo.com/v1/forecast` | `models=jma_gsm`, variables `*_previous_day1..3` | `f1_*`, `f2_*`, `f3_*` |
| Live weather | `api.open-meteo.com/v1/forecast` | `models=jma_gsm,best_match`, `past_days`, `forecast_days=4` | `live_daily` |
| CAMS PM2.5 | `air-quality-api.open-meteo.com/v1/air-quality` | `domains=cams_global`, `pm2_5` | `cams_pm25` (baseline + lagged feature) |

All free, no key. Licences: Open-Meteo data is CC BY 4.0. ERA5 and CAMS are Copernicus data
("Generated using Copernicus Climate Change Service / Atmosphere Monitoring Service information").

`models=era5` is set on purpose. The archive's default `best_match` mixes ERA5-Land (for
temperature and humidity) with ERA5 (for boundary-layer height) and, for the most recent days, an
operational model. One fixed product is easier to reason about.

## Variables and units

Every request uses `timezone=Asia/Bangkok` and `wind_speed_unit=ms`.

| Column | Open-Meteo variable | Daily rule | Unit |
|---|---|---|---|
| `w_temp`, `f{h}_temp` | `temperature_2m` | mean | °C |
| `w_rh`, `f{h}_rh` | `relative_humidity_2m` | mean | % |
| `w_wind`, `f{h}_wind` | `wind_speed_10m` | mean of the speed | m/s |
| `w_u`, `f{h}_u` | from speed and `wind_direction_10m` | mean of hourly u | m/s, positive = towards the east |
| `w_v`, `f{h}_v` | from speed and `wind_direction_10m` | mean of hourly v | m/s, positive = towards the north |
| `w_precip`, `f{h}_precip` | `precipitation` | daily total | mm/day |
| `w_blh` | `boundary_layer_height` | mean | m |
| `w_blh_min` | `boundary_layer_height` | minimum hour | m |
| `cams_pm25` | `pm2_5` | mean | µg/m³ |

Wind direction is where the wind comes **from** (0° = from the north). The components are
`u = -speed * sin(dir)` and `v = -speed * cos(dir)`, so a north wind has `v < 0`. We average u and v,
not the angle, because the mean of 350° and 10° must be "from the north", not "from the south".
The u/v pair also keeps the "where the air comes from" signal, for example smoke carried from the
north-west in the burning season.

The daily minimum of the boundary-layer height is a separate feature. A very shallow night-time
mixing layer traps smoke near the ground, and the daily mean hides it.

## Time zone

Days are local calendar days in Bangkok, the same day as the PM2.5 target. Asking the API for
`timezone=Asia/Bangkok` makes it return local hours, and the code checks `utc_offset_seconds == 25200`
in every answer. If the API ever fell back to UTC, every "day" would shift by 7 hours without any
visible error, so a wrong offset stops the build. Thailand has no daylight saving time, so every
day has exactly 24 hours.

## Daily aggregation rules

- **Enough hours or NaN.** A day needs at least 20 of its 24 hours for each variable, or the same
  share of a coarser grid (7 of 8 for 3-hourly data, 4 of 4 for 6-hourly). A mean built from only
  the night hours would be biased, so it becomes NaN instead.
- **Coarser grids.** The code finds the typical spacing between non-missing values (1, 3 or 6 h)
  and counts the required hours against that grid. In practice Open-Meteo returns every hour for
  ERA5, JMA and CAMS (JMA and CAMS are interpolated to hourly), so the hourly rule is the one that
  applies. The 3-hourly path is tested so that a change on the API side is handled correctly.
- **Precipitation total.** Open-Meteo precipitation is the amount in the preceding hour (mm/h).
  The daily total is `mean of the valid hours x 24`. On a complete day this is the plain sum. With a
  few hours missing, or on a 3-hourly grid with nulls in between, it does not undercount (a plain
  sum of 8 three-hourly values would give one third of the real total).
- **JMA precipitation in 6 h blocks.** JMA's precipitation arrives as 6-hour totals that Open-Meteo
  spreads evenly over the 6 hours (for example 3.0 mm in each of six hours = 18 mm). The blocks end
  at 00/06/12/18 UTC = 07/13/19/01 local, so one block crosses midnight. Spreading it evenly splits
  that block's rain fairly between the two days, and the daily sum stays correct.
- **No partial edge days.** Chunks start and end on local day boundaries, and the hourly chunks are
  joined before aggregating. The last days of ERA5 (about 5 days behind real time) are partial and
  become NaN; trailing days with no data at all are dropped from the file.

## Leakage: observed weather vs issued forecasts

The model issues its forecast at midnight at the end of day `t` and predicts days `t+1 .. t+3`.

- **`w_*` (ERA5)** is the weather that really happened on that date, rebuilt after the fact.
  It is only safe for dates on or before `t`. Using ERA5 for the target day would give the model
  perfect knowledge of the weather it is trying to forecast, which no real forecast has.
- **`f{h}_*` (JMA GSM previous runs)** is the forecast as it was issued. Open-Meteo's docs define
  `*_previous_day1` as "the value predicted 24 hours before valid time" and `_previous_day2` as
  48 hours before (`_previous_day0` is the latest run). So for every hour `T` of date `d`,
  `f{h}` comes from a run started at or before `T - 24h*h`.

For target day `t+h`, the hours run from 00:00 to 23:00 local on day `t+h`. Subtracting `24h*h`
gives run start times between 00:00 and 23:00 local on day `t`: every run used was started on
day `t` or earlier, before the midnight issue time. That is why the feature table may use `f{h}_*`
only at `target_date`, and only for the matching `h`.

One small caveat: JMA output appears some hours after the run starts. The 12 UTC run (19:00 local)
is normally published before midnight Bangkok time, so it was available at the issue time. The
argument above guarantees the run *started* before the issue time; it does not prove that each
run's files were *published* before midnight. At worst the late hours of the target day use a run
published around the issue time, never one from a later day.

A check with the Single Runs API (to match each hour to one named run) was inconclusive: values
were close to the 06 and 12 UTC runs of the day before but not identical, probably because of
different grid points or interpolation. The definition above is from Open-Meteo's documentation. A
weaker but useful check from our own data: the error of `f{h}_temp` against ERA5 grows with `h`
(numbers below), which is what forecasts made earlier should show.

JMA has no boundary-layer height, so BLH is only a lagged feature (ERA5, `w_blh` up to day `t`).

## CAMS: the stitched-lead caveat

The Air Quality API keeps, for every past hour, the value from the most recent CAMS run.
`pm2_5_previous_day1..N` exists in the API but is empty for this domain. So the archived CAMS value
for day `t+3` was issued around day `t+3`, with a 0 to 1 day lead, not 72 hours before.

This makes the comparison generous to CAMS at 48 and 72 h. If our model beats this archived CAMS at
those horizons, it would also beat a fair 72 h CAMS forecast. If CAMS wins there, the comparison was
unfair to our model. `cams_pm25` is used as the CAMS baseline at the target date and as a lagged
feature (day `t` and before), never as a target-day input to our models.

## Live data vs training data

`weather.live_daily(city, past_days=30)` and `cams.live_daily(city, past_days=30)` return one row per
day from `today - past_days` to `today + 3` (Bangkok dates), with the same columns as the processed
files.

| | Training | Live |
|---|---|---|
| `w_temp .. w_precip` | ERA5 reanalysis | JMA GSM past hours from the Forecast API (the first hours of recent runs, close to an analysis) |
| `w_blh`, `w_blh_min` | ERA5 | Forecast API `best_match` BLH (JMA has none) |
| `f{h}_*` for date `d` | JMA run at least `h` days before `d` | the latest JMA run, copied to `f1`, `f2` and `f3` |
| `cams_pm25` | archived, stitched latest runs | past days: same archive; today .. today+3: the current CAMS forecast |

- `w_*` is filled only for days before today; today and later are NaN (not yet observed).
- `f{h}_*` is filled only for today .. today+3; past days are NaN.
- Live `f{h}` uses a fresher run than training did (a 0 to 3 day lead instead of at least `h` days),
  so live forecasts get slightly better weather inputs than in the backtest. That is the safe
  direction: the backtest does not promise more than the live app can deliver.
- Live `w_*` is not ERA5, so small level differences (for example in BLH or rain) are possible. The
  model uses these only as lagged context, so the effect should be small, but it is a real
  train/serve difference.
- Live calls are not cached.

## Data gaps

**ERA5 boundary-layer height is missing for the first half of 2024.** Open-Meteo's archive returns
null BLH from 2024-01-01 07:00 to 2024-07-01 06:00 local time (that is 00:00 UTC on 1 January to
23:00 UTC on 30 June 2024). We checked `era5`, `era5_seamless`, `ecmwf_ifs` and `best_match`:
all null. With the 20-hour rule, `w_blh` and `w_blh_min` are NaN on **183 days (2024-01-01 to
2024-07-01) for every city, 1,281 rows in total**. Other ERA5 variables are complete. This hits the
**2024 burning season**, which is inside the backtest window. We do not fill the gap with another
source; the models must handle NaN in these two lagged features (LightGBM does this natively).

**JMA previous runs.** `f1` and `f2` start on 2019-01-01, `f3` on 2019-01-05 (the archive's day-3
values start at 2019-01-04 07:00, and a day needs 20 hours). This is earlier than the 2021-06 that
was first assumed: Open-Meteo says JMA GSM is archived from 2018.

**JMA archive gap in April 2026.** Some lead/variable pairs are null upstream from 2026-04-14 to
2026-04-27: `f1_rh` and `f3_temp` (14 days), `f2_precip` and `f3_precip` (13 days), in all seven
cities (98 rows have at least one NaN; no row is all NaN). This is inside the backtest window and
the 2026 burning season. It is left as NaN.

**CAMS** for Thailand starts at 2022-08-04 07:00 local; the first full day is **2022-08-05** in all
seven cities. Rows from 2022-07-01 to 2022-08-04 are NaN.

## Coverage (download of 2026-11-04)

| File | Rows | Dates | Coverage |
|---|---|---|---|
| `weather_past.parquet` | 20,013 | 2019-01-01 .. 2026-10-29 | 100% of days for all variables except the BLH gap above |
| `weather_fc.parquet` | 20,055 | 2019-01-01 .. 2026-11-04 | 100% of days for `f1`/`f2` in 2019-2025; `f3` from 2019-01-05; a short gap in April 2026 (below) |
| `cams_daily.parquet` | 11,116 | 2022-07-01 .. 2026-11-04 | 100% of days from 2022-08-05 (2022: 81% of the requested days) |

Lead check on our own data (all cities, 2019-01 .. 2026-10, mean absolute difference from ERA5
after removing each city-and-month mean bias, because JMA's 0.5° grid and ERA5's 0.25° grid differ
in level):

| | `f1` | `f2` | `f3` |
|---|---|---|---|
| temperature (°C) | 0.75 | 0.77 | 0.80 |
| wind speed (m/s) | 0.33 | 0.35 | 0.36 |
| relative humidity (%) | 5.36 | 5.43 | 5.51 |
| precipitation (mm/day) | 3.56 | 3.54 | 3.60 |

The error grows with the lead for temperature, wind and humidity, as expected from older runs.
Daily rain is hard at every lead, so its differences are within noise. `f1` and `f3` are different
values on more than 99% of days for the continuous variables, so the archive really holds
different runs per lead. Without the bias correction the raw differences are about 1.68 °C for
temperature at all three leads: that is mostly a fixed level difference between the two models.

## API calls used

Open-Meteo's free tier is for non-commercial use, under 10,000 calls per day. A request with many
variables or a long date range counts as several calls. Every uncached request is followed by a
1-second pause; retries and backoff are in `thaiair.http`, and a body that is not JSON is retried
after 30 to 60 seconds (this happened once with the slow previous-runs server).

| Step | HTTP requests |
|---|---|
| ERA5: 7 cities x 8 yearly chunks | 56 |
| JMA previous runs: 7 cities x 16 half-year chunks, plus 2 failed or retried | 114 |
| CAMS: 7 cities x 18 three-month chunks | 126 |
| Probing the APIs by hand | about 31 |
| Live functions, tried once each on real data | 2 |

Half-year chunks are used for JMA because one year of 15 variables takes about 30 seconds on the
server, and a 5-year request timed out. Daily updates after the first build need only the last
chunk per city and file.

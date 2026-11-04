# Thai Air Forecast

**Can a small model I own beat the professional CAMS (Copernicus) air-quality forecast?** This repo forecasts daily mean PM2.5 for 7 Thai cities 1, 2 and 3 days ahead, and tests that question on three burning seasons of held-out data.

![Chiang Mai smog episode: observed PM2.5 vs LightGBM vs CAMS](reports/figures/smog_episode.png)

<!-- RESULTS:headline -->
On 20,754 paired forecasts, at h=1 LightGBM MAE is 3.4 µg/m³ vs 9.8 for CAMS (skill +65%); at h=3 LightGBM MAE is 5.1 µg/m³ vs 9.9 for CAMS (skill +48%). In the burning season (Jan-Apr) at h=1: 5.3 vs 15.2 (skill +65%).
<!-- /RESULTS -->

## Why

Every year from January to April, northern and central Thailand fill with smoke. Farmers burn crop waste, forests burn, and the air stays still. Chiang Mai often has some of the worst air in the world during these months. Fine dust (PM2.5) goes deep into the lungs. On bad days Thai health agencies tell people to stay inside and some schools close.

A good forecast for the next few days lets people plan: keep children inside, buy masks, move an outdoor event. A professional forecast already exists. The Copernicus Atmosphere Monitoring Service (CAMS) runs a global chemistry and weather model and publishes PM2.5 forecasts for every point on Earth.

I want to know whether a **small model I own** can do better for these cities, using only free public data: city ground stations, archived weather forecasts, and CAMS itself as an input. The model runs on a laptop CPU, trains in minutes and costs nothing to serve. The alternative is to rent or depend on a big professional service. If the small model wins, owning it is worth it. If it loses, that is also a useful answer, and this README says so.

## Setup

- **Target:** daily mean PM2.5 (µg/m³) for each city, per local calendar day (Asia/Bangkok). A city's value is the median of a few long-running Air4Thai stations.
- **Issue time:** the end of day `t`. The model may use everything observed up to and including day `t`.
- **Horizons:** `h = 1, 2, 3` days, so the target is day `t + h` (24, 48 and 72 h ahead).
- **Cities (7):** Bangkok, Chiang Mai, Chiang Rai, Nan, Khon Kaen, Nakhon Ratchasima, Hat Yai.
- **Backtest:** rolling origin. At the start of each month the models are refit on every day whose target was already observed, then forecast that month. Nothing from the future is used.
- **Test window:** October 2023 to June 2026, which holds three burning seasons (2024, 2025, 2026).
- **Models:** two simple baselines (persistence: "tomorrow = today"; seasonal average: "what this week of the year usually looks like"), raw CAMS, CAMS with a per-city bias correction, LightGBM with quantiles (p10/p50/p90), and a small PyTorch quantile network.
- **Alerts:** an "unhealthy day" is a daily mean above 37.5 µg/m³ (Thai AQI orange or worse).

**An honest caveat about CAMS.** The free CAMS archive (via Open-Meteo) keeps only the most recent run for each past hour. So the archived CAMS value for day `t + 3` was issued around day `t + 3`, not 72 hours before. In this comparison CAMS always has a lead of about 0 to 1 day, while my models must forecast a true 24, 48 or 72 hours ahead. At h = 2 and h = 3 the comparison favours CAMS. A win there is a conservative result. A loss there may be partly due to the head start. See [docs/sources.md](docs/sources.md).

## Results

All numbers are on held-out months (rolling origin, Oct 2023 to Jun 2026). Every model is scored on the same rows: the (city, horizon, issue date) cases where every model and CAMS have a forecast and the truth is known. Full tables, by city too: [reports/results.md](reports/results.md).

### 1. Error by horizon

<!-- RESULTS:by_horizon -->
| Model | MAE h=1 | MAE h=2 | MAE h=3 | RMSE h=1 | RMSE h=2 | RMSE h=3 |
|:---|---:|---:|---:|---:|---:|---:|
| Persistence | 4.1 | 5.9 | 7.0 | 7.1 | 10.1 | 11.7 |
| Seasonal average | 7.9 | 8.0 | 8.0 | 12.8 | 12.9 | 12.9 |
| CAMS† | 9.8 | 9.9 | 9.9 | 15.5 | 15.5 | 15.5 |
| CAMS bias-corrected | 8.0 | 8.0 | 8.0 | 12.2 | 12.2 | 12.2 |
| LightGBM | **3.4** | **4.6** | **5.1** | **5.8** | **7.5** | **8.5** |
| Small neural net | 3.6 | 4.9 | 5.6 | 6.2 | 8.2 | 9.6 |

µg/m³, lower is better, best in bold. Paired rows per horizon: h=1: 6,928, h=2: 6,918, h=3: 6,908.

† CAMS is the Open-Meteo archive, a stitched series of the most recent CAMS run, so its real lead time is about 0-1 day at every horizon. Our models forecast a true 24/48/72 h ahead, so at h=2 and h=3 the comparison favours CAMS (see docs/sources.md). CAMS bias-corrected is built on the same archived value, so it shares this head start.
<!-- /RESULTS -->

### 2. Burning season vs the rest of the year

<!-- RESULTS:by_season -->
| Model | Burning h=1 | Burning h=2 | Burning h=3 | Rest h=1 | Rest h=2 | Rest h=3 |
|:---|---:|---:|---:|---:|---:|---:|
| Persistence | 6.4 | 9.6 | 11.5 | 2.8 | 3.8 | 4.4 |
| Seasonal average | 12.6 | 12.7 | 12.7 | 5.3 | 5.3 | 5.3 |
| CAMS | 15.2 | 15.2 | 15.2 | 6.8 | 6.8 | 6.9 |
| CAMS bias-corrected | 11.1 | 11.1 | 11.2 | 6.2 | 6.2 | 6.2 |
| LightGBM | **5.3** | **7.2** | **8.2** | **2.4** | **3.1** | **3.4** |
| Small neural net | 5.7 | 8.1 | 9.3 | 2.5 | 3.1 | 3.5 |

MAE µg/m³, best in bold. Burning season = target day in Jan, Feb, Mar, Apr. Paired rows at h=1: burning 2,498, rest 4,430. Mean observed PM2.5: burning 38.1, rest 16.9 µg/m³.
<!-- /RESULTS -->

![MAE by horizon, burning season vs rest](reports/figures/error_by_horizon.png)

### 3. Skill vs CAMS

<!-- RESULTS:skill -->
| Model | h=1 burning | h=1 rest | h=2 burning | h=2 rest | h=3 burning | h=3 rest |
|:---|---:|---:|---:|---:|---:|---:|
| Persistence | **+58%** | **+59%** | **+36%** | **+44%** | **+24%** | **+35%** |
| Seasonal average | **+17%** | **+22%** | **+17%** | **+22%** | **+16%** | **+22%** |
| CAMS† | ref (15.2) | ref (6.8) | ref (15.2) | ref (6.8) | ref (15.2) | ref (6.9) |
| CAMS bias-corrected | **+27%** | **+10%** | **+27%** | **+10%** | **+26%** | **+10%** |
| LightGBM | **+65%** | **+65%** | **+52%** | **+55%** | **+46%** | **+50%** |
| Small neural net | **+62%** | **+64%** | **+47%** | **+54%** | **+38%** | **+49%** |

Skill = 1 - MAE(model) / MAE(CAMS) on the same rows. Positive (bold) = better than CAMS. The CAMS row shows its own MAE in µg/m³.

† CAMS is the Open-Meteo archive, a stitched series of the most recent CAMS run, so its real lead time is about 0-1 day at every horizon. Our models forecast a true 24/48/72 h ahead, so at h=2 and h=3 the comparison favours CAMS (see docs/sources.md). CAMS bias-corrected is built on the same archived value, so it shares this head start. A positive skill at h=2/3 is therefore a conservative result; a negative one may be partly due to the lead-time gap.
<!-- /RESULTS -->

![Skill vs CAMS by city](reports/figures/skill_by_city.png)

**Where CAMS wins** (generated, so the losses cannot be left out):

<!-- RESULTS:cams_wins -->
- **Persistence**: CAMS wins in 0 of 6 horizon-season groups and 0 of 21 city-horizon groups.
- **Seasonal average**: CAMS wins in 0 of 6 horizon-season groups and 6 of 21 city-horizon groups.
- **CAMS bias-corrected**: CAMS wins in 0 of 6 horizon-season groups and 6 of 21 city-horizon groups.
  - Chiang Mai, h=3: skill -7% (MAE 10.1 vs CAMS 9.4)
  - Chiang Mai, h=2: skill -7% (MAE 10.1 vs CAMS 9.4)
  - Chiang Mai, h=1: skill -7% (MAE 10.0 vs CAMS 9.4)
  - Chiang Rai, h=3: skill -5% (MAE 10.0 vs CAMS 9.6)
  - Chiang Rai, h=2: skill -4% (MAE 10.0 vs CAMS 9.5)
  - Chiang Rai, h=1: skill -4% (MAE 10.0 vs CAMS 9.5)
- **LightGBM**: CAMS wins in 0 of 6 horizon-season groups and 0 of 21 city-horizon groups.
- **Small neural net**: CAMS wins in 0 of 6 horizon-season groups and 0 of 21 city-horizon groups.

Read this with the CAMS caveat: CAMS is the Open-Meteo archive, a stitched series of the most recent CAMS run, so its real lead time is about 0-1 day at every horizon. Our models forecast a true 24/48/72 h ahead, so at h=2 and h=3 the comparison favours CAMS (see docs/sources.md). CAMS bias-corrected is built on the same archived value, so it shares this head start.
<!-- /RESULTS -->

### 4. Unhealthy-day alerts

<!-- RESULTS:alerts -->
| Model | h | Events | Alerts | Precision | Recall | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Persistence | 1 | 1,112 | 1,112 | 0.79 | 0.79 | 0.79 |
| Seasonal average | 1 | 1,112 | 1,354 | 0.58 | 0.71 | 0.64 |
| CAMS | 1 | 1,112 | 532 | 0.70 | 0.34 | 0.46 |
| CAMS bias-corrected | 1 | 1,112 | 844 | 0.71 | 0.54 | 0.61 |
| LightGBM | 1 | 1,112 | 1,108 | 0.84 | 0.84 | **0.84** |
| Small neural net | 1 | 1,112 | 1,238 | 0.79 | 0.88 | 0.83 |
| Persistence | 2 | 1,112 | 1,112 | 0.70 | 0.70 | 0.70 |
| Seasonal average | 2 | 1,112 | 1,354 | 0.58 | 0.71 | 0.64 |
| CAMS | 2 | 1,112 | 532 | 0.70 | 0.34 | 0.46 |
| CAMS bias-corrected | 2 | 1,112 | 843 | 0.71 | 0.54 | 0.61 |
| LightGBM | 2 | 1,112 | 1,110 | 0.77 | 0.77 | **0.77** |
| Small neural net | 2 | 1,112 | 1,202 | 0.74 | 0.80 | 0.77 |
| Persistence | 3 | 1,111 | 1,111 | 0.65 | 0.65 | 0.65 |
| Seasonal average | 3 | 1,111 | 1,355 | 0.58 | 0.71 | 0.64 |
| CAMS | 3 | 1,111 | 532 | 0.70 | 0.34 | 0.46 |
| CAMS bias-corrected | 3 | 1,111 | 842 | 0.71 | 0.54 | 0.61 |
| LightGBM | 3 | 1,111 | 1,102 | 0.75 | 0.75 | **0.75** |
| Small neural net | 3 | 1,111 | 1,187 | 0.72 | 0.77 | 0.74 |

An event is an observed daily mean above 37.5 µg/m³ (Thai AQI orange or worse). An alert is a p50 forecast above the same line. Precision = share of alerts that were real; recall = share of events that got an alert. Best F1 per horizon in bold.
<!-- /RESULTS -->

### 5. Are the intervals honest?

<!-- RESULTS:coverage -->
| Model | h | n | Coverage | Burning | Rest | Width |
|:---|---:|---:|---:|---:|---:|---:|
| Persistence | 1 | 6,928 | 83% | 80% | 84% | 13.6 |
| Persistence | 2 | 6,918 | 82% | 78% | 84% | 19.6 |
| Persistence | 3 | 6,908 | 82% | 78% | 84% | 23.3 |
| Seasonal average | 1 | 6,928 | 75% | 75% | 76% | 23.4 |
| Seasonal average | 2 | 6,918 | 75% | 75% | 76% | 23.4 |
| Seasonal average | 3 | 6,908 | 75% | 75% | 75% | 23.4 |
| CAMS | 1 | 6,928 | 78% | 84% | 75% | 30.1 |
| CAMS | 2 | 6,918 | 78% | 84% | 75% | 30.1 |
| CAMS | 3 | 6,908 | 78% | 84% | 75% | 30.1 |
| CAMS bias-corrected | 1 | 6,928 | 76% | 81% | 74% | 25.8 |
| CAMS bias-corrected | 2 | 6,918 | 76% | 81% | 74% | 25.9 |
| CAMS bias-corrected | 3 | 6,908 | 76% | 81% | 74% | 25.9 |
| LightGBM | 1 | 6,928 | 82% | 82% | 82% | 11.5 |
| LightGBM | 2 | 6,918 | 82% | 80% | 83% | 15.3 |
| LightGBM | 3 | 6,908 | 81% | 80% | 82% | 17.0 |
| Small neural net | 1 | 6,928 | 80% | 82% | 78% | 12.5 |
| Small neural net | 2 | 6,918 | 75% | 76% | 75% | 15.0 |
| Small neural net | 3 | 6,908 | 72% | 67% | 75% | 14.4 |

A p10-p90 interval should contain the truth 80% of the time. Below 80% = overconfident, above = too wide. Width = mean p90 - p10 in µg/m³ (narrower is better only if coverage holds).
<!-- /RESULTS -->

### Takeaways

- **Yes, the small model beats CAMS, everywhere.** LightGBM's error is 3.4 µg/m³ vs 9.8 for CAMS at 24 h (skill +65%) and 5.1 vs 9.9 at 72 h (+48%). It wins in all 21 city-horizon pairs and in both seasons, even though CAMS gets a 0-1 day head start at 48/72 h.
- **But raw CAMS is a low bar for city stations.** CAMS forecasts a ~40 km grid cell from emission inventories, and here it reads low (mean bias −6.0 µg/m³). In Chiang Mai in March 2024 it stayed around 20-45 while the stations read 100-160 (figure above). Even "tomorrow = today" beats it by 58% at 24 h. So the honest yardstick is **persistence**: LightGBM beats it by 16% at 24 h and 27% at 72 h. The gain grows with the horizon because today's value says less about day 3.
- **The burning season is where it matters and where it is hardest.** Errors are about 2.2× larger in Jan-Apr (LightGBM 5.3 vs 2.4 µg/m³ at 24 h). LightGBM is still best there, and beats persistence by 17% at 24 h and 29% at 72 h.
- **The cheap fix is not enough.** Bias-correcting CAMS per city cuts its burning-season error by 27%, but it is still worse than persistence: a scale correction cannot fix CAMS missing the timing of local smoke. The small neural net is close to LightGBM at 24 h (3.6 vs 3.4) but falls behind at 72 h and its 72 h intervals are too narrow (72% coverage). LightGBM wins every metric, so the app serves it. (The net first scored 6.7 at 24 h, worse than persistence; predicting the *change* from today instead of the level fixed it. See [deep.py](src/thaiair/models/deep.py).)
- **Alerts:** LightGBM catches 84% of unhealthy days at 24 h with 84% precision (F1 0.84), and 75% / 75% at 72 h. CAMS catches only 34% of them, because it under-forecasts smog.
- **The intervals are honest:** LightGBM's p10-p90 band holds the truth 81-82% of the time at every horizon, in and out of the burning season (target 80%), thanks to the conformal calibration step in [lgbm.py](src/thaiair/models/lgbm.py).
- **Hardest city: Chiang Rai at 72 h** (MAE 6.7, skill vs CAMS only +30%). My guess, not tested: its smoke often comes across the Myanmar and Laos borders, which the model's local station and weather inputs cannot see. Fire-hotspot data would be the test (see Limitations).

## Try it

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[deep,viz,demo,dev]"
```

On Windows set `PYTHONPATH=src;.` (Linux/macOS: `src:.`), then:

```bash
# API: GET /health, GET /cities, GET /forecast?city=chiang_mai, GET /backtest?city=chiang_mai&horizon=1
python -m uvicorn app.api:app --port 8000
curl "http://127.0.0.1:8000/forecast?city=chiang_mai"

# Dashboard: live 3-day map for the 7 cities, and the backtest against CAMS
python -m streamlit run app/dashboard.py

# Tests and lint
python -m pytest
ruff check .
```

Docker (API only, CPU, LightGBM): `docker build -t thai-air-forecast .` then `docker run -p 8000:8000 thai-air-forecast`.

The live forecast pulls today's data from Air4Thai, Open-Meteo and CAMS (all free, no API key). The app loads the trained models from `models/`. If they are missing it falls back to a clearly labelled persistence forecast, so it always starts.

A [GitHub Action](.github/workflows/daily-forecast.yml) runs the live forecast for all 7 cities every morning (08:30 Bangkok time) and saves it as a downloadable artifact. It commits nothing back to the repo.

A `Makefile` wraps the common commands (`make data backtest train report api dashboard test`).

## Reproduce from scratch

```bash
python -m thaiair.pm25       # Air4Thai daily files 2019-2026 + recent hourly -> data/processed/pm25_daily.parquet
python -m thaiair.weather    # ERA5 past weather + archived JMA forecasts -> weather_past / weather_fc
python -m thaiair.cams       # CAMS PM2.5 archive -> cams_daily.parquet
python -m thaiair.evaluate   # rolling-origin backtest of every model -> reports/backtest_predictions.parquet
python -m thaiair.train      # final models on all data -> models/*.joblib + manifest.json
python -m thaiair.report     # reports/results.md + reports/figures/*.png (add --fill-readme to update this file)
```

Raw downloads are cached in `data/raw/` (git-ignored), so a re-run makes no new calls. Everything here runs on a laptop CPU, no GPU.

## Data

All sources are free and need no API key. Details, coverage and licences: [docs/sources.md](docs/sources.md). Cleaning: [docs/cleaning.md](docs/cleaning.md). Weather and CAMS: [docs/weather_cams.md](docs/weather_cams.md).

| What | Source | Licence / terms |
|---|---|---|
| Ground-truth PM2.5 (daily 2019 to Jun 2026, hourly for the last ~2 months) | Thai Pollution Control Department, [Air4Thai](https://air4thai.pcd.go.th) | No licence stated; Thai government data. Credited here; raw files are not redistributed, `thaiair.pm25` downloads them. |
| Past weather (observed) | ERA5 reanalysis via [Open-Meteo](https://open-meteo.com) | CC BY 4.0 (Open-Meteo); Copernicus licence (ERA5) |
| Weather forecasts as they were issued | JMA GSM via the Open-Meteo Previous Runs API | CC BY 4.0 |
| CAMS PM2.5 forecast (the baseline to beat) | Copernicus Atmosphere Monitoring Service via the Open-Meteo Air Quality API | CC BY 4.0 + "Generated using Copernicus Atmosphere Monitoring Service information" |

Weather data by Open-Meteo.com. Contains Copernicus Atmosphere Monitoring Service information.

## Limitations

- **Daily, not hourly.** There is no public multi-year hourly archive of Thai station data (the hourly service keeps only about 2 months), so the model forecasts the daily mean. Thai AQI for PM2.5 uses the 24 h mean anyway, but hourly peaks are not covered.
- **The CAMS comparison is tilted towards CAMS** at 48 and 72 h, because the archive is a stitched series with a 0 to 1 day lead (see above). The bias-corrected CAMS baseline uses the same archived value, so it shares the head start: at h = 2 and 3 it is not a model you could run live with the same inputs.
- **A city is a few stations.** Each city value is the median of 1 to 9 stations. Chiang Rai, Nan, Khon Kaen, Nakhon Ratchasima and Hat Yai have one station each, so one broken sensor moves the whole city.
- **No fire data.** NASA FIRMS hotspot counts would likely help in the burning season, but the API needs a key, so they are not used.
- **A gap in July to August 2026.** The yearly PCD file ends on 30 June 2026 and the hourly service starts on 1 September 2026. It falls in the rainy season and outside the backtest.
- **Coarse weather forecasts.** The only model with a multi-year forecast archive is JMA's global model, which is coarse for mountain valleys like Chiang Mai. Boundary-layer height is only in ERA5, so it is used as a lagged (observed) feature only.
- **Gaps in the weather archive.** Open-Meteo has no boundary-layer height for 1 Jan to 1 Jul 2024 (every archive model is empty), which is the whole 2024 burning season, and a few JMA forecast variables are missing for 14 to 27 April 2026. These are left as missing values (LightGBM handles them natively), not filled in.
- **One burning season for every city.** "Burning season" means Jan-Apr everywhere. That fits the north well; Bangkok's worst months start earlier (Dec), and Hat Yai's haze comes from Indonesian fires around Aug-Oct.
- **Live features differ slightly from training.** The live app uses the freshest forecast run and the forecast API's recent weather, while training used archived runs and ERA5. The live forecast may be a little better or worse than the backtest suggests.
- **One country, three seasons.** Three burning seasons is a small sample of bad years. The results may not hold in a very different year or another country.

## Layout

```
src/thaiair/   data (pm25, weather, cams), dataset + features, baselines, backtest, metrics, report, live
src/thaiair/models/   LightGBM quantile model, small PyTorch net, saved-model registry
app/           FastAPI service + Streamlit dashboard
models/        trained models used by the app
reports/       results.md, figures, backtest predictions
docs/          architecture (the interface contract), data sources, cleaning, weather + CAMS notes
tests/         unit tests (no network, synthetic data; run in CI)
.github/       CI and the daily forecast Action
LEARN.md       the ideas behind the project, explained simply, with a quiz and exercises
```

Built with AI assistance (Claude Code).

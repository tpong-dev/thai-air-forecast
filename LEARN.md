# Learn from this project

This file is for me, as a learner. It explains the ideas behind the repo in simple words, with tiny examples, and points to the file where each idea lives. Then it gives a 60-second pitch, likely interview questions, a quiz, three hands-on exercises, and a set of design problems to re-derive on my own.

Numbers in the examples are made up to show the idea. Real results are in [reports/results.md](reports/results.md).

## Part 1. The key ideas

### 1. Leakage in time series

**Idea.** Leakage means the model sees, during training or testing, something it could not know at the moment the forecast is made. The scores then look great, and the live forecast is much worse.

In time series the usual leak is the future. Examples:
- Using the *observed* weather of the target day as a feature. At issue time (the end of day `t`) nobody knows tomorrow's real wind. You only know the *forecast* of tomorrow's wind.
- A rolling mean that is centred (uses days after `t`) instead of trailing (ends on `t`).
- Filling a missing value with the next day's value (back-fill).
- Shuffling rows before a train/test split, so the model trains on March 12 and is tested on March 11.

**Tiny example.** Tomorrow's PM2.5 depends strongly on tomorrow's wind. If you give the model tomorrow's *observed* wind, it learns "low wind = smog" perfectly and the error drops by a lot. But at 08:00 today the real wind of tomorrow does not exist yet. Live, you only have a forecast of it, which can be wrong by 1 to 2 m/s. The backtest score was a lie.

**In this repo.** The rule is written in [docs/architecture.md](docs/architecture.md), section 3: pm25, observed weather (`w_*`) and CAMS may only come from dates `<= issue_date`; weather *forecasts* (`f{h}_*`) are allowed for the target day because they were issued at least `h` days earlier. [src/thaiair/features.py](src/thaiair/features.py) builds features only from backward shifts and trailing windows. [tests/test_features.py](tests/test_features.py) proves it: it changes every value after the issue date and checks that no feature moves.

### 2. Backtesting with a rolling origin

**Idea.** To test a forecaster honestly, replay history as if you were living through it. Stand at a point in time (the "origin"), train only on the past, forecast the next stretch, score it, then move the origin forward and repeat.

**Tiny example.** Test month = March 2025. Train on every row whose *target* day is before 1 March 2025. Forecast every issue date in March. Then move to April: retrain on everything before 1 April, forecast April. The training set grows each month ("expanding window").

Why "target before the month" and not "issue before the month"? A row issued on 28 February with h = 3 has its label on 3 March. That label was not known on 1 March, so it must not be in the training set.

**In this repo.** [src/thaiair/backtest.py](src/thaiair/backtest.py) (`run`, monthly refit), driven by [src/thaiair/evaluate.py](src/thaiair/evaluate.py). The test window (Oct 2023 to Jun 2026) is in [src/thaiair/config.py](src/thaiair/config.py).

### 3. Persistence baseline

**Idea.** The simplest forecast: "tomorrow will be like today". For PM2.5 it is strong, because smog episodes last for days. Any model must beat it, or it is useless.

**Tiny example.** Today Chiang Mai is 80 µg/m³. Persistence says 80 for tomorrow, 80 for the day after. If tomorrow is 75, the error is only 5. At 3 days ahead the weather has often changed, so persistence gets worse fast.

**In this repo.** `Persistence` in [src/thaiair/baselines.py](src/thaiair/baselines.py).

### 4. Seasonal baseline (climatology)

**Idea.** "What does this time of year usually look like in this city?" It knows nothing about today, but it knows that March is bad and July is clean.

**Tiny example.** Over past years, mid-March in Chiang Mai averaged 70 µg/m³. The seasonal baseline forecasts 70 for every mid-March day, smog or not. It is weak at h = 1 but does not get worse with the horizon, so at long horizons it can beat persistence.

**In this repo.** `SeasonalAverage` in [src/thaiair/baselines.py](src/thaiair/baselines.py) (a window of +/- 7 days of the year, per city).

### 5. Quantiles, pinball loss and interval coverage

**Idea.** A single number hides how sure the model is. A quantile forecast gives a range: p10 (the value the truth should fall below 10% of the time), p50 (the median, used as the point forecast) and p90. The p10 to p90 band should contain the truth 80% of the time.

You train a quantile with the **pinball loss**. For quantile `q` and error `e = y - prediction`:
- if `e > 0` (forecast too low), loss = `q * e`
- if `e < 0` (forecast too high), loss = `(q - 1) * e`

**Tiny example.** For p90 (q = 0.9): forecasting 10 too low costs 0.9 × 10 = 9; forecasting 10 too high costs 0.1 × 10 = 1. So the p90 model learns to aim high, until only about 10% of truths are above it.

**Coverage** checks the promise: count the share of days where p10 <= y <= p90. If it is 60%, the model is overconfident. If it is 95%, the band is too wide to be useful. Always report coverage *and* width together.

**In this repo.** [src/thaiair/models/lgbm.py](src/thaiair/models/lgbm.py) (LightGBM with the quantile objective, plus a conformal adjustment that widens the band using the latest held-out errors), [src/thaiair/models/deep.py](src/thaiair/models/deep.py) (pinball loss written by hand), `coverage` in [src/thaiair/metrics.py](src/thaiair/metrics.py).

### 6. Skill scores

**Idea.** A raw error (MAE 8 µg/m³) is hard to judge alone. A skill score compares it with a reference on the *same rows*:

`skill = 1 - MAE(model) / MAE(reference)`

Skill > 0: better than the reference. Skill = 0: the same. Skill < 0: worse.

**Tiny example.** MAE of LightGBM = 9, MAE of CAMS = 12. Skill = 1 - 9/12 = +25%. If LightGBM were 13, skill = -8%.

"Same rows" matters. CAMS only exists from September 2022, and a model may skip days with missing data. If each model is scored on its own rows, one model can look better just because it skipped the hard days.

**In this repo.** `skill` and `paired` in [src/thaiair/metrics.py](src/thaiair/metrics.py); the tables and the "where CAMS wins" list in [src/thaiair/report.py](src/thaiair/report.py).

### 7. Alert precision and recall

**Idea.** People care less about "MAE 8" and more about "will tomorrow be unhealthy?". Turn the forecast into an alert: alert if p50 > 37.5 µg/m³. Then score it like a classifier:
- **Precision**: of the alerts you sent, how many were real unhealthy days?
- **Recall**: of the real unhealthy days, how many got an alert?
- **F1**: one number that balances both (harmonic mean).

**Tiny example.** 10 real unhealthy days. The model alerts on 12 days, and 8 of those were real. Precision = 8/12 = 0.67. Recall = 8/10 = 0.80. Lower the alert line and recall goes up, but so do false alarms (precision goes down).

**In this repo.** `alert_scores` in [src/thaiair/metrics.py](src/thaiair/metrics.py); the threshold `UNHEALTHY_PM25` in [src/thaiair/config.py](src/thaiair/config.py).

## Part 2. What I must be able to explain in an interview

### The 60-second pitch

> Every year from January to April, smoke from crop and forest burning makes northern Thailand's air some of the worst in the world. I built a system that forecasts daily PM2.5 for seven Thai cities one, two and three days ahead, and I asked one question: can a small model I own beat the professional Copernicus CAMS forecast? I used only free public data: government station readings, weather forecasts as they were actually issued, and CAMS itself. I tested everything with a rolling-origin backtest over three burning seasons, retraining every month on the past only, and scored every model on exactly the same rows. LightGBM with quantiles TODO(lead: result in one clause). I was careful about one unfair point: the free CAMS archive has a 0 to 1 day lead, so at 48 and 72 hours the comparison favours CAMS, and I say so in the results. It is served as a FastAPI service and a Streamlit map, and a GitHub Action runs the forecast every morning.

### Likely follow-up questions (and what a good answer covers)

1. **Why daily and not hourly?** No public multi-year hourly archive (only about 2 months). Thai AQI for PM2.5 uses the 24 h mean anyway.
2. **How do you know there is no leakage?** The written rule, features built only from trailing windows, the test that changes future data and checks features do not move, and weather forecasts taken from archived runs that were at least `h` days old.
3. **Why rolling origin and not a random split?** Neighbouring days are strongly correlated; a random split lets the model "see" the answer from the day before and after. Rolling origin copies how the model is used live.
4. **Why is persistence so hard to beat at 1 day?** PM2.5 is very autocorrelated: smog episodes last for days. The model's value shows when the weather changes (rain, wind, boundary layer).
5. **Is the CAMS comparison fair?** Not fully. Explain the stitched archive. A win at 48/72 h is conservative; a loss may be partly caused by the head start.
6. **Why LightGBM and not a deep model?** A few thousand rows per city, tabular features. Trees are strong here, fast on CPU and easy to explain. The small neural net is there to test that claim, not assume it.
7. **How did you get honest intervals?** Quantile loss, then a conformal correction on the most recent held-out stretch, then checking coverage on the test months.
8. **Which errors matter most?** Missing an unhealthy day is worse than a false alarm for health; show the precision/recall trade-off and how the threshold changes it.
9. **What happens in production when a data source is down?** The app falls back to persistence and labels it; missing stations are handled by the city median; long outages are not hidden behind stale values.
10. **What would you add next?** Fire hotspots (NASA FIRMS), more stations, a fair CAMS archive (real 72 h runs from the Copernicus Atmosphere Data Store), hourly forecasts once an archive builds up.

## Part 3. Quiz (10 questions)

1. At the end of day `t`, you forecast day `t + 2`. Which of these can be a feature: (a) observed wind on day `t + 2`, (b) the JMA wind forecast for day `t + 2` from a run made on day `t`, (c) PM2.5 on day `t`?

<details><summary>Answer</summary>(b) and (c). (a) is the observed future: leakage. (b) was known at issue time because the run was made on day t.</details>

2. In a monthly rolling-origin backtest for April, can a row issued on 30 March with h = 3 be in the training set?

<details><summary>Answer</summary>No. Its target day is 2 April, which was not observed on 1 April. Training rows need target_date before the fold starts.</details>

3. Persistence has MAE 6 at h = 1 and 11 at h = 3. Why does it get so much worse?

<details><summary>Answer</summary>Smog episodes last a few days, so today is a good guess for tomorrow, but over 3 days the weather (wind, rain, mixing height) often changes and the episode starts or ends.</details>

4. Model MAE = 10, CAMS MAE = 8 on the same rows. What is the skill vs CAMS?

<details><summary>Answer</summary>1 - 10/8 = -0.25, so -25%: the model is worse than CAMS.</details>

5. Why must skill be computed on paired rows?

<details><summary>Answer</summary>Otherwise each model is scored on different days. A model that skipped hard days (outages during smog, or the years before CAMS existed) would look better than it is.</details>

6. For the p10 quantile, what does the pinball loss charge for a forecast 10 too low, and 10 too high?

<details><summary>Answer</summary>Too low (y above forecast): 0.1 × 10 = 1. Too high: 0.9 × 10 = 9. So the p10 model aims low.</details>

7. Your p10-p90 band has 62% coverage. What does that mean and what would you do?

<details><summary>Answer</summary>The band is too narrow: the model is overconfident (target is 80%). Widen it, for example with a conformal correction from recent held-out errors, and check coverage again on new data.</details>

8. 20 real unhealthy days. The model sends 25 alerts; 15 are correct. Precision and recall?

<details><summary>Answer</summary>Precision = 15/25 = 0.60. Recall = 15/20 = 0.75.</details>

9. Why does the archived CAMS series favour CAMS at h = 3?

<details><summary>Answer</summary>Open-Meteo keeps only the latest run for each past hour, so the value for day t+3 came from a run made around day t+3 (0-1 day lead), not one made 72 h earlier.</details>

10. Why is boundary-layer height used only as a lagged feature?

<details><summary>Answer</summary>It is only available from ERA5 reanalysis (observed), not from the archived JMA forecasts. Observed values of the target day would be leakage, so only past days can be used.</details>

## Part 4. Hands-on exercises

### Exercise 1. Break leakage on purpose

Goal: see what a leak looks like, so I can spot one later.

1. In a copy of `make_features` (do not change the real one), add the *observed* target-day weather: shift `w_wind`, `w_blh` and `w_precip` by `-h` so day `t + h` values land on the issue row.
2. Run a short backtest: `python -m thaiair.evaluate --models persistence,cams,lgbm --quick` (synthetic data), or the real one if data is downloaded.
3. Compare MAE with and without the leak. Expect the leaked model to be "too good", especially at h = 3, where the real forecast is weakest.
4. Run `tests/test_features.py` against your copy and watch the leakage test fail. That test is the guard.

### Exercise 2. Move the alert threshold

Goal: see the precision/recall trade-off.

1. Load `reports/backtest_predictions.parquet` with pandas and keep the LightGBM rows at h = 1.
2. Keep the event fixed (y > 37.5), but alert when p50 > T for T from 20 to 60 in steps of 2. Also try "alert when p90 > 37.5".
3. For each T compute precision and recall (`thaiair.metrics.alert_scores` or by hand) and plot recall (x) against precision (y).
4. Decide which T you would use for a school that wants to miss at most 1 unhealthy day in 10. What does it cost in false alarms?

### Exercise 3. Add NASA FIRMS fire counts

Goal: add a real new feature without leaking.

1. Get a free FIRMS MAP_KEY and download VIIRS hotspots for Thailand and its neighbours (fires in Myanmar and Laos matter for the north).
2. For each city count hotspots per day within, say, 100 km and 300 km.
3. Add them as **lagged** features only (day `t` and before; a 3-day sum is a good start). Fire counts for the target day are not known in advance.
4. Re-run the backtest and look at burning-season skill vs CAMS, by city. Does it help Chiang Mai and Nan more than Bangkok? Write down whether it helped, even if it did not.

## Part 5. Re-derive the design

This is the Socratic re-derive method. Each problem below is one design decision, in the order the system would naturally be built. Read the problem and its first constraint, and write down your answer *before* reading the next constraint. Each extra constraint is meant to break the naive answer. There are no answers here on purpose: try to rebuild the design yourself, then compare with the code.

You can also ask Claude to run a live re-derive session on this repo ("run a re-derive session on thai-air-forecast"): it asks one problem at a time, pushes back on your answer with the next constraint, and only then shows where the real design lives.

### Problem 1. What exactly should we predict?

We want to warn people about bad air in Thai cities.
- **First constraint:** the warning must be useful a few days ahead, not just for the next hour.
- **Then:** the public hourly station archive keeps only about 2 months; the multi-year archive is daily.
- **Then:** a city has several stations, and some stations go offline for days.
- **Then:** the official Thai AQI for PM2.5 is defined on a 24 h mean.

### Problem 2. How do we test without peeking at the future?

We have seven years of daily data and want an honest error number.
- **First constraint:** the model will be used live, every day, on days it has never seen.
- **Then:** neighbouring days are strongly correlated, and smog episodes last for days.
- **Then:** a row issued at the end of a month with h = 3 has its label in the next month.
- **Then:** we want results across several burning seasons, not one lucky or unlucky year, and the CAMS archive only starts in late 2022.

### Problem 3. What does "good" mean?

The model has an MAE of 9 µg/m³. Is that good?
- **First constraint:** the number must mean something to someone who does not know the data.
- **Then:** PM2.5 is very autocorrelated, so "tomorrow = today" is already strong at 1 day.
- **Then:** the burning season has much bigger values and errors than the rainy season, so one overall number hides the part that matters.
- **Then:** the real competitor is a professional forecast (CAMS), and not every model has a forecast on every day.

### Problem 4. Weather features without cheating

Weather drives smog: wind, rain and mixing height decide whether smoke builds up or clears.
- **First constraint:** at issue time the target day's weather has not happened yet.
- **Then:** live weather forecasts exist, but we need *past* forecasts as they were issued, to train and test fairly.
- **Then:** the only model with a multi-year forecast archive is a coarse global model, and it has no boundary-layer height.
- **Then:** the live app gets a fresher forecast run than the one the backtest used.

### Problem 5. Uncertainty and honest intervals

A single number does not tell people how sure we are.
- **First constraint:** we want a range that holds the truth about 80% of the time.
- **Then:** tree models trained on quantile loss are usually overconfident out of sample.
- **Then:** errors are skewed: big misses are mostly on the high side, during smog peaks.
- **Then:** the baselines need intervals too, so coverage can be compared on equal terms.

### Problem 6. Turning numbers into alerts

People want a yes/no: "is tomorrow unhealthy?"
- **First constraint:** there is an official line (37.5 µg/m³, Thai AQI orange).
- **Then:** a missed unhealthy day and a false alarm do not cost the same.
- **Then:** unhealthy days are rare outside the burning season and very common inside it.

### Problem 7. A fair fight with CAMS

The headline question is "can my model beat CAMS?".
- **First constraint:** compare on the same days, cities and horizons.
- **Then:** the free CAMS archive keeps only the most recent run, so its "3-day forecast" was issued about 0 to 1 day ahead.
- **Then:** CAMS has a known bias for these cities, and a simple correction might be all you need.
- **Then:** CAMS is also allowed as an input to our model. When is that fair, and when is it leakage?

### Problem 8. Serving when data or models are missing

The forecast runs every morning on live APIs.
- **First constraint:** the API and dashboard must always start and return something.
- **Then:** a station or an API can be down for hours or days.
- **Then:** saved model files may be missing (fresh clone, Docker image without models).
- **Then:** the live features are built from slightly different sources than the training features.

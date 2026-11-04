# Data cleaning: PM2.5 ground truth

This page explains how the PM2.5 target (`data/processed/pm25_daily.parquet`) is built from raw
station data, what every cleaning rule does, and how many values each rule changed on the real data.
Weather and CAMS cleaning are described in [weather_cams.md](weather_cams.md).

Code: `src/thaiair/pm25.py`. Rebuild everything with:

```bash
python -m thaiair.pm25
```

All numbers on this page come from the run on 2026-11-04 (data up to 2026-11-03).

## Sources

| Source | What it is | Period used |
|---|---|---|
| PCD yearly daily files, `https://air4thai.net/daily_download/PM2.5/{year}.xlsx` | Pollution Control Department archive. One sheet: rows = days, columns = station IDs, values = 24 h mean in µg/m³. | 2019-01-01 to 2026-06-30 |
| Air4Thai hourly history service, `getHistoryData.php` | Hourly values per station. It only keeps about the last 2 months. | 2026-09-01 to yesterday |

Raw files are cached in `data/raw/pcd_daily/{year}.xlsx` and `data/raw/a4t_hourly/{station}_{YYYY-MM}.json`
(git-ignored). A month of hourly data is cached for good once it ended more than 2 days ago,
because the service deletes old data and the cache is then the only copy. The current month is
downloaded again on every run. Delete `data/raw/pcd_daily/2026.xlsx` when PCD publishes a newer
version of the yearly file.

Only the 16 stations listed in `src/thaiair/config.py` are kept (see [sources.md](sources.md) for why
these stations).

**TLS note.** The Air4Thai servers send an incomplete certificate chain, so plain `requests` fails
certificate checks. `pm25.py` checks these certificates with the operating system's trust store
(through `truststore`) for Air4Thai URLs only. TLS checks stay on.

## Time zone

The data is in Thai local time (Asia/Bangkok, UTC+7, no daylight saving time). A "day" is a local
calendar day. Dates are stored as naive timestamps at local midnight (`datetime64[ns]`, no time
zone), the same as every other processed file in this project. For hourly data, the hours
00:00 to 23:00 of a date belong to that date. "Today" is always today in Bangkok, whatever the
time zone of the computer.

## Parsing the yearly files

All eight files (2019 to 2026) were checked by hand. Problems found and how the parser handles them:

| Quirk | Where | Handling |
|---|---|---|
| Sheet name changes ("PM2.5", "Data", "DATA"); a second sheet lists the stations | all years | Use the first sheet. |
| Missing values written as text: `n/a`, `N/A` | all years | Text that is not a number becomes NaN. |
| Footnote rows in Thai at the bottom (for example "N/A = no data", the legal limit) and blank rows; the 2024 sheet has about 4,000 empty rows | all years | Only rows whose first cell is a date are kept. |
| Values are whole numbers until early 2023, then one decimal | 2019-2023 | No change (both are valid means). |
| Duplicate date: 2021-06-30 appears twice and 2021-06-29 is missing | 2021 | The rows are in order, so the first copy is 2021-06-29 with a typo. The date is rebuilt from the row position. **1 row relabelled.** If such a fix is not clearly safe, the parser drops every row of the duplicated date instead. |
| Header not on the first row, station IDs with odd case or spaces, the same station twice, rows dated in another year | none in the real files | Handled anyway (tested): the header row is found by its "Date" cell, IDs are trimmed and upper-cased, duplicate columns are merged, other-year rows are dropped. |
| Sentinel numbers (-1, 0, 999) | none found | Covered by the range rule below. Smallest real value: 2.0, largest: 586 (Mae Sai 73T, March 2023, not one of our stations). |

For our 16 stations the yearly files mark **1,499 station-days** as missing (N/A or blank). These
stay NaN.

## From hourly values to a daily mean

Used for data after the last yearly file, and by the live app.

1. Hourly values below 0 or above 1,000 µg/m³ count as missing (these are error codes, not air).
   On the real data (24,576 hourly rows, Sept to Nov 2026): 928 hours were empty (`null`), none were
   out of range.
2. A daily mean needs **at least 18 valid hours out of 24 (75 %)**. This is the completeness rule the
   US EPA uses for 24 h PM2.5. With fewer hours, one missing night or rush hour can bias the mean a
   lot. On the real data **41 station-days** failed this rule (26 had no data at all, 15 had 3 to 17
   hours) and became NaN. 983 station-days passed.
3. The mean is rounded to 0.1 µg/m³, like the PCD files.
4. Today is never included, because today is not complete yet.

Where both sources have the same station-day, the PCD yearly file wins (it is PCD's checked archive;
the hourly service is preliminary). This counts even when the PCD value is N/A.

## Cleaning rules for station daily values

The rules are conservative on purpose. Burning-season peaks in the north are real and are exactly
what the forecast must get right, so a high value alone is never removed. Values that a rule removes
become NaN; nothing is filled in.

| Rule | Definition | Why | Values changed |
|---|---|---|---|
| Out of range | value < 0 or > 1,000 µg/m³ | Physically impossible / above the instrument range: an error code. | **0** |
| Stuck sensor | the same value on 7 or more days in a row | A working monitor's 24 h mean changes every day. A frozen logger repeats its last value. | **0** |
| Isolated spike | more than 4 times the larger neighbour day **and** more than 100 µg/m³ above it, with both neighbours present | Catches a one-day glitch. Real smoke episodes build up and fade over several days. | **0** |

The thresholds were chosen by looking at the real data, not to remove values:

* **Stuck.** The longest honest run of identical values is 6 days (61T in August 2022, 14 µg/m³, a
  clean rainy-season week with whole-number values). A threshold of 6 would remove 6 values, 5 would
  remove 46, and 4 would remove 214, all of them low, rainy-season whole numbers that look real.
  7 days is the first threshold that removes nothing real.
* **Spike.** Even a much looser rule (3 times and +50 µg/m³) finds nothing. At 2 times and +25 µg/m³
  it finds 4 days (Bangkok 52T 2019-02-13 and 05T 2022-03-09, Hat Yai 2026-01-01, Chiang Rai
  2024-05-02). They look like real one-day events (the Hat Yai one is the New Year fireworks night),
  so a rule that loose would delete real air. At 1.8 times, Bangkok 02T and 50T both jump on
  2024-04-11: two stations agreeing is a real event, not a sensor fault. The highest value kept is
  374 µg/m³ at Chiang Rai 57T on 2023-03-27, during the severe March 2023 haze (Mae Sai reached 586
  the same day).

So on the real data **no value was changed by a cleaning rule**. The rules stay in the code as a
guard for future data and for the live feed.

## City aggregation

City daily PM2.5 = **median** of the city's stations that reported that day. `n_stations` = how many
stations that was (0 when none reported).

* The median keeps one roadside or faulty Bangkok station from pulling the city value. For Chiang Mai
  (2 stations) it is the mean of the two; for one-station cities it is the station value.
* Bangkok normally has 9 stations: 9 on 2,255 days, 7-8 on 522 days, 5-6 on 25 days, 0 on 62 days
  (the July-August 2026 gap).
* The output has one row per city for **every** day from 2019-01-01 to the last day with data
  (2026-11-03): 7 cities x 2,864 days = 20,048 rows. Days without data are rows with NaN, not
  missing rows.

Output files:

| File | Columns |
|---|---|
| `data/processed/pm25_daily.parquet` | `city` (str), `date` (datetime64[ns], naive local midnight), `pm25` (float), `n_stations` (int64) |
| `data/processed/pm25_station_daily.parquet` | `station`, `date`, `pm25`, `source` (`pcd_daily` or `a4t_hourly`); 44,791 rows |

## Coverage

Share of days with a city value:

| city | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bangkok | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 80% |
| chiang_mai | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 80% |
| chiang_rai | 99% | 99% | 99% | 88% | 100% | 100% | 98% | 78% |
| hat_yai | 55% | 97% | 100% | 99% | 99% | 100% | 96% | 79% |
| khon_kaen | 99% | 100% | 96% | 76% | 96% | 99% | 100% | 79% |
| nakhon_ratchasima | 51% | 99% | 100% | 100% | 100% | 99% | 99% | 79% |
| nan | 100% | 99% | 99% | 99% | 100% | 99% | 94% | 72% |

2026 counts days up to 2026-11-03. Notes:

* **Hat Yai and Nakhon Ratchasima 2019:** their stations (44T, 47T) start reporting in mid-June 2019.
* **Khon Kaen 2022:** station 46T has no data from 2022-07-13 to the end of the year (88 days).
* **2026 for every city: the July-August gap.** The last yearly file ends on 2026-06-30 and the
  hourly service only goes back to 2026-09-01, so **2026-07-01 to 2026-08-31 (62 days) is missing for
  all cities**. It falls in the rainy season (low PM2.5) and outside the backtest window
  (2023-10-01 to 2026-06-30). It will close when PCD publishes the full 2026 yearly file.

## Why the target is never imputed

`pm25` is what the models are trained and scored on. If missing days were filled (by interpolation,
a neighbour station or a seasonal average), the model would be partly scored on numbers we made up.
Filled values are smooth and easy to predict, so they would make every model, and especially
persistence, look better than it is. They would also hide the days when a station was down. So a
missing day stays NaN: rows with NaN `y` are dropped from training and scoring, and feature code
decides by itself how to handle NaN in lagged inputs.

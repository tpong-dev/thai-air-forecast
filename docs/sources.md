# Data sources, access and licences

Checked by hand on 2026-11-04, before any code was written. All sources are free and need **no API key**.

| What | Source | Endpoint | Key? | Coverage checked | Licence / terms |
|---|---|---|---|---|---|
| Ground-truth PM2.5, daily, multi-year | Thai Pollution Control Department (PCD), Air4Thai "Archived Data → daily" files | `https://air4thai.net/daily_download/PM2.5/{year}.xlsx` | No | 2019-01-01 to 2026-06-30 (one sheet: rows = dates, columns = station IDs, values = 24 h mean µg/m³) | No licence is stated on Air4Thai. PCD is a Thai government agency, and Thai government open data (data.go.th) is published under the "Open Data Common" licence. This repo credits PCD and does **not** redistribute the raw files: `python -m thaiair.pm25` downloads them. Only derived city-level daily means and model outputs are committed. |
| Ground-truth PM2.5, hourly, recent | Air4Thai history service | `https://air4thai.com/forweb/getHistoryData.php?stationID=..&param=PM25&type=hr&sdate=..&edate=..&stime=00&etime=23` | No | Only the latest ~2 months (from 2026-09-01 on the check date; earlier dates return empty) | As above. Used to extend the series past June 2026 and to feed the live app. |
| Station list and coordinates, live readings | Air4Thai | `https://air4thai.pcd.go.th/services/getNewAQI_JSON.php` | No | 173 stations (GROUND / BKK / MOBILE) | As above. |
| Past weather (observed) | ERA5 reanalysis via Open-Meteo Historical Weather API | `https://archive-api.open-meteo.com/v1/archive` | No | 1940 on, ~5-day delay; includes boundary-layer height | Open-Meteo data: CC BY 4.0 ("Weather data by Open-Meteo.com"). ERA5 is Copernicus Climate Change Service data under the Copernicus licence. |
| Weather **forecasts as they were issued** (leakage-free features) | JMA GSM via Open-Meteo Previous Runs API (`*_previous_day1..3`) | `https://previous-runs-api.open-meteo.com/v1/forecast?models=jma_gsm` | No | Wind, rain, humidity, temperature from at least 2021-06. No boundary-layer height (that is only in ERA5, so BLH is used as a lagged feature only). Other models (GFS, ECMWF, ICON) are only archived from 2024. | CC BY 4.0 (Open-Meteo). |
| Live weather forecast | Open-Meteo Forecast API | `https://api.open-meteo.com/v1/forecast` | No | `past_days` up to 92, 16 days ahead | CC BY 4.0. |
| Baseline to beat: CAMS | Copernicus Atmosphere Monitoring Service global forecast via Open-Meteo Air Quality API | `https://air-quality-api.open-meteo.com/v1/air-quality` | No | Thailand PM2.5 from about 2022-09 (2022-08 and earlier return null). `pm2_5_previous_dayN` exists but is empty, so the archive is a stitched series of the latest runs. | CC BY 4.0 (Open-Meteo) + Copernicus licence: "Generated using Copernicus Atmosphere Monitoring Service information". |
| Fire hotspots (optional) | NASA FIRMS | n/a | Free MAP_KEY | **Not used**: it needs a key, and the brief said to skip it unless asked. | n/a |

Usage stays well inside Open-Meteo's free, non-commercial tier (under 10,000 calls/day per their terms). Raw responses are cached in `data/raw/` (git-ignored), so a re-run makes no new calls.

## What this means for the design

1. **Daily, not hourly.** Multi-year hourly station data is not publicly downloadable (only ~2 months), so the model forecasts the **daily mean** for day +1, +2 and +3 (24/48/72 h). That is also what the Thai AQI uses for PM2.5.
2. **A gap in July to August 2026.** The yearly file ends on 30 June 2026 and the hourly service starts on 1 September 2026. The gap is in the rainy season (low PM2.5) and falls outside the backtest window.
3. **CAMS is evaluated generously.** Open-Meteo keeps only the most recent CAMS run for each past hour, so the archived CAMS value for day `t+3` was issued around day `t+3`, not 72 h earlier. Our model must forecast 72 h ahead, while the CAMS it is compared with had a 0 to 1 day lead. If the model beats this CAMS at 48/72 h, it would also beat a fair 72 h CAMS forecast. If CAMS wins at 48/72 h, the comparison is unfair to our model.
4. **Weather features use forecasts as they were issued.** For target day `t+h`, the model sees the JMA forecast from a run at least `h` days old (`previous_day{h}`), never the observed weather of the target day.

## Station choice

City value = median of the stations listed in `src/thaiair/config.py` (each had ≥ 88% daily coverage in every year 2020-2026, except 46T Khon Kaen at 76% in 2022):

- Bangkok: 02T, 03T, 05T, 12T, 50T, 52T, 54T, 59T, 61T (general and roadside ground stations)
- Chiang Mai: 35T (Chang Phueak), 36T (Si Phum)
- Chiang Rai: 57T (Wiang, Mueang)
- Nan: 67T (Nai Wiang, Mueang)
- Khon Kaen: 46T · Nakhon Ratchasima: 47T · Hat Yai: 44T

Out-of-town stations (for example Mae Sai 73T, the Mae Mo power-plant stations in Lampang) were left out on purpose, so each series measures the city itself.

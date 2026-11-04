"""Streamlit dashboard: live 3-day PM2.5 forecast map and the backtest against CAMS.

    python -m streamlit run app/dashboard.py        (with PYTHONPATH=src;.)

Calls `thaiair.live.forecast` in-process (no API server needed) and reads
reports/backtest_predictions.parquet for the backtest tab. Every failure is shown as a
message, never a stack trace, so the page stays usable when a source is down.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):  # so `thaiair.*` and `app.*` import under `streamlit run`
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.api import backtest_view, get_backtest_path, load_backtest  # noqa: E402
from thaiair import config, live  # noqa: E402

# Thai AQI colours (PCD): very good = blue, good = green, moderate = yellow, then orange and red.
BAND_COLORS: dict[str, tuple[int, int, int]] = {
    "very good": (59, 204, 255),
    "good": (146, 208, 80),
    "moderate": (255, 255, 0),
    "starts to affect health": (255, 162, 0),
    "affects health": (240, 70, 70),
}
NO_DATA_COLOR = (150, 150, 150)
CREDITS = (
    "PM2.5 observations: Pollution Control Department (PCD), Air4Thai · "
    "Weather: [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0) · "
    "Generated using Copernicus Atmosphere Monitoring Service information (CAMS, via Open-Meteo). "
    "Forecasts are experimental and not an official air-quality warning."
)


# --------------------------------------------------------------------------- data (cached)


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(city: str) -> dict[str, Any]:
    """One city's forecast, or {"error": ...}; errors are values so st.cache_data can store them."""
    try:
        return live.forecast(city)
    except (live.LiveDataError, live.UnknownCityError) as exc:
        return {"city": city, "name": config.CITIES[city].name, "error": str(exc)}
    except Exception as exc:  # unexpected bug: keep the page alive, show a short message
        return {"city": city, "name": config.CITIES[city].name, "error": f"forecast failed ({type(exc).__name__})"}


@st.cache_data(ttl=600, show_spinner=False)
def get_backtest() -> pd.DataFrame | None:
    try:
        return load_backtest(get_backtest_path())
    except Exception:  # missing or unreadable file -> the tab explains how to create it
        return None


# --------------------------------------------------------------------------- forecast tab


def day1(fc: dict[str, Any]) -> dict[str, Any] | None:
    return next((f for f in fc.get("forecasts", []) if f["horizon"] == 1), None)


def map_rows(forecasts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fc in forecasts:
        city = config.CITIES[fc["city"]]
        d1 = day1(fc)
        band = d1["aqi_band"] if d1 else None
        p50 = d1["p50"] if d1 else None
        rows.append(
            {
                "name": city.name,
                "lat": city.lat,
                "lon": city.lon,
                "color": list(BAND_COLORS.get(band, NO_DATA_COLOR)),
                "label": f"{p50:.0f} µg/m³ ({band})" if p50 is not None else "no forecast",
            }
        )
    return pd.DataFrame(rows)


def render_map(forecasts: list[dict[str, Any]]) -> None:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_rows(forecasts),
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
        stroked=True,
        get_radius=25000,
        radius_min_pixels=8,
        pickable=True,
    )
    view = pdk.ViewState(latitude=13.2, longitude=100.9, zoom=4.6)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{name}\nDay +1: {label}"}))
    st.caption(
        " · ".join(f"{band}: ≤ {upper:g}" for upper, band in config.THAI_AQI_PM25_BANDS[:-1]) + " µg/m³ (24 h mean)"
    )


def forecast_chart(fc: dict[str, Any]) -> alt.LayerChart:
    """Last 14 observed days, then the p10-p90 band, p50 line and CAMS points for days +1..+3."""
    obs = pd.DataFrame(fc["recent"]).assign(kind="observed")
    fut = pd.DataFrame(fc["forecasts"])
    obs["date"], fut["date"] = pd.to_datetime(obs["date"]), pd.to_datetime(fut["date"])
    y = alt.Y("pm25:Q", title="PM2.5 daily mean (µg/m³)")
    observed = alt.Chart(obs).mark_line(point=True, color="#444").encode(x=alt.X("date:T", title=None), y=y)
    band = alt.Chart(fut).mark_area(opacity=0.25, color="#1f77b4").encode(x="date:T", y="p10:Q", y2="p90:Q")
    median = alt.Chart(fut).mark_line(point=True, color="#1f77b4").encode(x="date:T", y=alt.Y("p50:Q"))
    cams = (
        alt.Chart(fut.dropna(subset=["cams"]))
        .mark_point(shape="diamond", size=90, color="#d62728", filled=True)
        .encode(x="date:T", y="cams:Q", tooltip=["date:T", "cams:Q"])
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [config.UNHEALTHY_PM25]}))
        .mark_rule(strokeDash=[4, 4], color="orange")
        .encode(y="y:Q")
    )
    return (observed + band + median + cams + threshold).properties(height=340)


def forecast_table(fc: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(fc["forecasts"])
    df["unhealthy"] = df["unhealthy"].map({True: "yes", False: "no"})
    return df[["date", "horizon", "p10", "p50", "p90", "aqi_band", "unhealthy", "cams"]].rename(
        columns={"horizon": "day +", "aqi_band": "Thai AQI band", "unhealthy": "> 37.5?", "cams": "CAMS"}
    )


def forecast_tab() -> None:
    with st.spinner("Fetching live data for 7 cities..."):
        forecasts = [get_forecast(c) for c in config.CITIES]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Today (day +1 from yesterday's data), Thai AQI band")
        render_map(forecasts)
    with right:
        keys = list(config.CITIES)
        city = st.selectbox(
            "City", keys, index=keys.index("chiang_mai"), format_func=lambda k: config.CITIES[k].name, key="fc_city"
        )
        fc = next(f for f in forecasts if f["city"] == city)
        if "error" in fc:
            st.error(f"No forecast for {fc['name']}: {fc['error']}")
            return
        st.markdown(f"**{fc['name']}**, issued for {fc['issue_date']} · model: `{fc['model']}`")
        if fc["model"] == live.FALLBACK_LABEL:
            st.info(
                "Saved models were not found, so this is a persistence forecast (every day = the last observed day)."
            )
        for w in fc.get("warnings", []):
            st.warning(w)
        st.altair_chart(forecast_chart(fc), width="stretch")
        st.caption("Blue: model median and p10-p90 band · red diamonds: CAMS · dashed: 37.5 µg/m³ alert line")
        st.dataframe(forecast_table(fc), hide_index=True, width="stretch")
        st.caption(f"Generated {fc['generated_at']}")


# --------------------------------------------------------------------------- backtest tab


def backtest_chart(view: dict[str, Any]) -> alt.Chart:
    df = pd.DataFrame(view["series"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    long = df.melt(id_vars="target_date", value_vars=["actual", "p50", "cams"], var_name="series", value_name="pm25")
    long["series"] = long["series"].map({"actual": "actual", "p50": view["model"], "cams": "CAMS"})
    return (
        alt.Chart(long.dropna())
        .mark_line()
        .encode(
            x=alt.X("target_date:T", title="target date"),
            y=alt.Y("pm25:Q", title="PM2.5 daily mean (µg/m³)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["actual", view["model"], "CAMS"], range=["#444", "#1f77b4", "#d62728"]),
            ),
            tooltip=["target_date:T", "series:N", alt.Tooltip("pm25:Q", format=".1f")],
        )
        .properties(height=380)
    )


def backtest_tab() -> None:
    preds = get_backtest()
    if preds is None:
        st.info(
            "No backtest results yet. Run `python -m thaiair.evaluate` to create reports/backtest_predictions.parquet."
        )
        return
    keys = list(config.CITIES)
    c1, c2, c3, c4 = st.columns([1.2, 0.6, 1, 1.6])
    city = c1.selectbox(
        "City", keys, index=keys.index("chiang_mai"), format_func=lambda k: config.CITIES[k].name, key="bt_city"
    )
    horizon = c2.selectbox("Days ahead", list(config.HORIZONS), key="bt_h")
    models = sorted(m for m in preds["model"].unique() if m != "cams")
    if not models:
        st.warning("The backtest file has no model rows.")
        return
    default = models.index(live.SERVED_MODEL) if live.SERVED_MODEL in models else 0
    model = c3.selectbox("Model", models, index=default, key="bt_model")
    lo, hi = preds["target_date"].min().date(), preds["target_date"].max().date()
    window = c4.date_input("Target dates", (max(lo, date(hi.year, 1, 1)), hi), min_value=lo, max_value=hi, key="bt_win")
    if not isinstance(window, tuple) or len(window) != 2:
        st.info("Pick a start and an end date.")
        return
    try:
        view = backtest_view(preds, city, int(horizon), window[0], window[1], model)
    except LookupError as exc:
        st.warning(str(exc))
        return
    if not view["series"]:
        st.info("No backtest rows for this selection.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric(f"MAE {model}", f"{view['mae_model']:.1f}" if view["mae_model"] is not None else "n/a")
    m2.metric("MAE CAMS", f"{view['mae_cams']:.1f}" if view["mae_cams"] is not None else "n/a")
    m3.metric("Days compared", view["n"])
    st.altair_chart(backtest_chart(view), width="stretch")
    st.caption(
        "MAE in µg/m³ on the days where actual, model and CAMS all exist. The archived CAMS has a 0-1 day lead, "
        "so at days +2 and +3 the comparison favours CAMS (see docs/sources.md)."
    )


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Thai Air Forecast", layout="wide")
    st.title("Thai Air Forecast")
    st.caption("Daily mean PM2.5 for 7 Thai cities, 1-3 days ahead, next to the CAMS (Copernicus) forecast.")
    tab_fc, tab_bt = st.tabs(["Forecast", "Backtest"])
    with tab_fc:
        forecast_tab()
    with tab_bt:
        backtest_tab()
    st.divider()
    st.caption(CREDITS)


main()

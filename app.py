"""
Sri Lanka Rain Forecast: Ensemble Learning Dashboard
Run:  streamlit run app.py
Models and artefacts are produced by ensemble_rain_sri_lanka.ipynb (saved in ./models).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Sri Lanka Rain Forecast | Ensemble Learning", layout="wide",
                   initial_sidebar_state="expanded")

FAMILY_COLORS = {"Baseline": "#94a3b8", "Bagging": "#0f766e", "Boosting": "#ea580c",
                 "Stacking": "#1e3a8a", "Hybrid": "#7c3aed"}
SEASONS = {12: "NE Monsoon", 1: "NE Monsoon", 2: "NE Monsoon", 3: "First Inter-monsoon",
           4: "First Inter-monsoon", 5: "SW Monsoon", 6: "SW Monsoon", 7: "SW Monsoon",
           8: "SW Monsoon", 9: "SW Monsoon", 10: "Second Inter-monsoon", 11: "Second Inter-monsoon"}
METRIC_LABELS = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "f1": "F1-score",
                 "recall": "Recall", "precision": "Precision"}

# ------------------------------------------------------------------ styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, select, textarea, label, p, h1, h2, h3, h4, h5 {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.stApp { background: #f4f6f9; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1480px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* sidebar */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.brand { display: flex; align-items: center; gap: 10px; padding-bottom: 14px; border-bottom: 1px solid #eef0f3;
         margin-bottom: 14px; }
.brand .logo { width: 34px; height: 34px; border-radius: 9px; background: #0f4c81; color: #fff;
               display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }
.brand .name { font-weight: 700; font-size: 15px; color: #0f172a; line-height: 1.2; }
.brand .sub { font-size: 11.5px; color: #6b7280; }
.side-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
              color: #6b7280; margin: 18px 0 6px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] > label { padding: 7px 10px; border-radius: 8px; width: 100%;
                                                        margin: 0; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: #f1f5f9; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: #e8f0fa; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: #0f4c81; font-weight: 600; }

/* page header */
.page-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px;
             margin-bottom: 16px; }
.page-head h1 { font-size: 1.55rem; font-weight: 700; color: #0f172a; margin: 0; padding: 0; letter-spacing: -0.01em; }
.page-head .desc { color: #6b7280; font-size: 13.5px; margin-top: 4px; }
.page-head .meta { font-size: 12px; color: #6b7280; background: #fff; border: 1px solid #e5e7eb;
                   padding: 6px 12px; border-radius: 8px; }

/* KPI cards */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 16px; }
.kpi { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; }
.kpi .label { font-size: 12px; color: #6b7280; font-weight: 500; }
.kpi .value { font-size: 1.45rem; font-weight: 700; color: #0f172a; margin-top: 4px; line-height: 1.25; }
.kpi .sub { font-size: 12px; color: #6b7280; margin-top: 2px; }
.kpi .bar { height: 3px; width: 36px; border-radius: 2px; margin-bottom: 10px; }
@media (max-width: 1000px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .card-title) {
    background: #fff; border: 1px solid #e5e7eb !important; border-radius: 12px !important;
}
.card-title { font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 2px; }
.card-sub { font-size: 12px; color: #6b7280; margin-bottom: 6px; }

/* verdict */
.verdict { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap;
           border-radius: 10px; padding: 12px 14px; margin-top: 4px; }
.verdict.rain { background: #eaf2fb; border: 1px solid #bcd3ee; }
.verdict.dry  { background: #fdf5e7; border: 1px solid #f3d9a4; }
.verdict .main { font-weight: 700; font-size: 15px; color: #0f172a; }
.verdict .pill { font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; background: #fff;
                 border: 1px solid #e5e7eb; color: #374151; margin-left: 6px; }
.pill.ok { color: #15803d; border-color: #bbf7d0; background: #f0fdf4; }
.pill.bad { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }

/* readings grid */
.readings { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.reading { background: #f8fafc; border: 1px solid #eef0f3; border-radius: 10px; padding: 10px 12px; }
.reading .k { font-size: 11.5px; color: #6b7280; }
.reading .v { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px; }
@media (max-width: 1000px) { .readings { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_meta():
    return json.load(open(ROOT / "models/meta.json"))


@st.cache_resource
def load_model(path):
    return joblib.load(ROOT / path)


@st.cache_data
def load_test_days():
    return pd.read_csv(ROOT / "models/test_days.csv", parse_dates=["date"])


@st.cache_data
def load_raw():
    return pd.read_csv(ROOT / "data/sri_lanka_weather.csv", parse_dates=["date"])


meta = load_meta()
days = load_test_days()
metrics = pd.DataFrame(meta["metrics"])
registry = meta["registry"]
model_names = list(metrics.sort_values("roc_auc", ascending=False).model)
city_info = days.groupby("city")[["climate_zone", "province", "latitude", "longitude"]].first()
cities = sorted(city_info.index)


# ------------------------------------------------------------------ helpers
def card(title, sub=None):
    box = st.container(border=True)
    box.markdown(f"<div class='card-title'>{title}</div>" + (f"<div class='card-sub'>{sub}</div>" if sub else ""),
                 unsafe_allow_html=True)
    return box


def page_header(title, desc):
    st.markdown(f"""<div class="page-head"><div><h1>{title}</h1><div class="desc">{desc}</div></div>
    <div class="meta">Train {meta['train_period']} &nbsp;|&nbsp; Test {meta['test_period']}</div></div>""",
                unsafe_allow_html=True)


def kpis(items):
    html = "".join(f"""<div class="kpi"><div class="bar" style="background:{c}"></div>
        <div class="label">{l}</div><div class="value">{v}</div><div class="sub">{s}</div></div>"""
                   for l, v, s, c in items)
    st.markdown(f"<div class='kpis'>{html}</div>", unsafe_allow_html=True)


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(t=10, b=10, l=10, r=10), legend_title_text="",
                      font=dict(family="Inter, sans-serif", size=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("""<div class="brand"><div class="logo">SL</div><div>
        <div class="name">Rain Forecast</div><div class="sub">Ensemble learning dashboard</div></div></div>""",
                unsafe_allow_html=True)
    page = st.radio("Navigation", ["Forecast", "Model performance", "Feature importance", "Data explorer", "About"],
                    label_visibility="collapsed")

best = metrics.sort_values("roc_auc", ascending=False).iloc[0]
best_recall = metrics.sort_values("recall", ascending=False).iloc[0]

# ================================================================== FORECAST
if page == "Forecast":
    with st.sidebar:
        st.markdown("<div class='side-label'>Forecast inputs</div>", unsafe_allow_html=True)
        model_choice = st.selectbox("Model", model_names,
                                    format_func=lambda m: f"{m} ({registry[m]['family']})")
        city = st.selectbox("City", cities, index=cities.index("Colombo"))
        mode = st.radio("Input mode", ["Replay a real day", "Custom weather"], horizontal=True)

        city_days = days[days.city == city].sort_values("date")
        if mode == "Replay a real day":
            d = st.date_input("Date (today)", value=city_days.date.iloc[-200].date(),
                              min_value=city_days.date.min().date(), max_value=city_days.date.max().date())
            row = city_days[city_days.date == pd.Timestamp(d)]
            if row.empty:
                st.warning("No record for that date; using the nearest day.")
                row = city_days.iloc[[np.argmin(np.abs(city_days.date - pd.Timestamp(d)))]]
            x = row[meta["features"]].copy()
            truth = int(row["rain_tomorrow"].iloc[0])
            sel_date = row.date.iloc[0]
            month = sel_date.month
        else:
            truth, sel_date = None, None
            base = city_days[meta["features"]].median(numeric_only=True)
            month = st.slider("Month", 1, 12, 6)
            tmax = st.slider("Max temperature (°C)", 10.0, 40.0, float(round(base.temperature_2m_max, 1)))
            tmin = st.slider("Min temperature (°C)", 5.0, 30.0, float(round(base.temperature_2m_min, 1)))
            rain = st.slider("Rain today (mm)", 0.0, 150.0, 2.0)
            rhours = st.slider("Hours of rain today", 0.0, 24.0, min(24.0, rain / 1.5))
            rain_y = st.slider("Rain yesterday (mm)", 0.0, 150.0, 2.0)
            rain7 = st.slider("Rain last 7 days (mm)", 0.0, 400.0, 20.0)
            hum = st.slider("Humidity (%)", 40.0, 100.0, float(round(base.relative_humidity_2m_mean)))
            pres = st.slider("Sea-level pressure (hPa)", 995.0, 1020.0, float(round(base.pressure_msl_mean, 1)))
            dp = st.slider("Pressure change vs yesterday (hPa)", -6.0, 6.0, 0.0)
            cloud = st.slider("Cloud cover (%)", 0.0, 100.0, float(round(base.cloud_cover_mean)))
            wind = st.slider("Max wind speed (km/h)", 0.0, 60.0, float(round(base.wind_speed_10m_max)))
            wdir = st.slider("Dominant wind direction (°)", 0, 359, 240)

            info = city_info.loc[city]
            tmean = (tmax + tmin) / 2
            dew = tmean - (100 - hum) / 5          # Lawrence approximation
            doy = pd.Timestamp(2024, month, 15).dayofyear
            vals = dict(base)
            vals.update(dict(
                city=city, climate_zone=info.climate_zone, province=info.province,
                monsoon_season=SEASONS[month], latitude=info.latitude, longitude=info.longitude,
                temperature_2m_max=tmax, temperature_2m_min=tmin, temperature_2m_mean=tmean,
                apparent_temperature_mean=tmean + 0.1 * (hum - 50), temp_range=tmax - tmin,
                precipitation_sum=rain, precipitation_hours=rhours, wind_speed_10m_max=wind,
                wind_gusts_10m_max=wind * 2.0, wind_dir_sin=np.sin(np.deg2rad(wdir)),
                wind_dir_cos=np.cos(np.deg2rad(wdir)), relative_humidity_2m_mean=hum,
                pressure_msl_mean=pres, cloud_cover_mean=cloud, dew_point_2m_mean=dew,
                rain_lag1=rain_y, rain_3d_sum=rain + rain_y + vals.get("rain_lag2", 0),
                rain_7d_sum=max(rain7, rain + rain_y), pressure_change=dp,
                month_sin=np.sin(2 * np.pi * month / 12), month_cos=np.cos(2 * np.pi * month / 12),
                doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25)))
            x = pd.DataFrame([vals])[meta["features"]]

    # ---- predictions
    p = float(load_model(registry[model_choice]["file"]).predict_proba(x)[0, 1])
    all_p = pd.DataFrame([{"model": m, "family": registry[m]["family"],
                           "p_rain": float(load_model(registry[m]["file"]).predict_proba(x)[0, 1])}
                          for m in model_names]).sort_values("p_rain")
    n_rain = int((all_p.p_rain >= 0.5).sum())
    info = city_info.loc[city]
    when = sel_date.strftime("%d %b %Y") if sel_date is not None else f"custom inputs, month {month}"

    page_header(f"Next-day rain forecast: {city}",
                f"{info.province} Province | {info.climate_zone} zone | {SEASONS[month]} | {when}")
    kpis([
        ("Chance of rain tomorrow", f"{p:.0%}", model_choice, "#0f4c81"),
        ("Forecast", "Rain" if p >= 0.5 else "Dry", "threshold 50%", "#0f4c81" if p >= 0.5 else "#d97706"),
        ("Model consensus", f"{n_rain} of {len(all_p)}", "models predict rain", "#7c3aed"),
        ("Actual next day", "-" if truth is None else ("Rain" if truth else "No rain"),
         "custom scenario" if truth is None else
         ("forecast correct" if (p >= 0.5) == bool(truth) else "forecast missed"),
         "#94a3b8" if truth is None else ("#15803d" if (p >= 0.5) == bool(truth) else "#b91c1c")),
    ])

    c1, c2 = st.columns([1, 1.35], gap="medium")
    with c1:
        with card("Probability of rain", f"{model_choice} | {registry[model_choice]['family']}"):
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100,
                number={"suffix": "%", "font": {"size": 46, "color": "#0f172a"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickfont": {"size": 11}},
                       "bar": {"color": "#0f4c81" if p >= 0.5 else "#d97706", "thickness": 0.3},
                       "bgcolor": "white", "borderwidth": 0,
                       "steps": [{"range": [0, 50], "color": "#fbf1dd"}, {"range": [50, 100], "color": "#e3edf8"}],
                       "threshold": {"line": {"color": "#111827", "width": 2}, "value": 50}}))
            st.plotly_chart(style_fig(gauge, 250))
            rainy = p >= 0.5
            pills = ""
            if truth is not None:
                ok = rainy == bool(truth)
                pills = (f"<span class='pill'>Actual next day: {'Rain' if truth else 'No rain'}</span>"
                         f"<span class='pill {'ok' if ok else 'bad'}'>{'Correct' if ok else 'Missed'}</span>")
            st.markdown(f"<div class='verdict {'rain' if rainy else 'dry'}'><span class='main'>"
                        f"{'Rain expected' if rainy else 'Likely dry'}</span><span>{pills}</span></div>",
                        unsafe_allow_html=True)
    with c2:
        with card("Model consensus", "Probability of rain tomorrow from every trained model (dashed line = 50%)"):
            fig = px.bar(all_p, x="p_rain", y="model", color="family", orientation="h",
                         color_discrete_map=FAMILY_COLORS, range_x=[0, 1],
                         labels={"p_rain": "P(rain tomorrow)", "model": ""})
            fig.add_vline(x=0.5, line_dash="dash", line_color="#374151", line_width=1)
            fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0))
            st.plotly_chart(style_fig(fig, 330))

    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        if sel_date is not None:
            with card("Recent rainfall", f"{city}: 14 days up to the selected date, plus the actual next day"):
                win = city_days[(city_days.date > sel_date - pd.Timedelta(days=14)) &
                                (city_days.date <= sel_date + pd.Timedelta(days=1))].copy()
                win["period"] = np.where(win.date == sel_date, "Selected day",
                                         np.where(win.date > sel_date, "Next day (actual)", "Previous days"))
                fig = px.bar(win, x="date", y="precipitation_sum", color="period",
                             color_discrete_map={"Previous days": "#9fb8d6", "Selected day": "#0f4c81",
                                                 "Next day (actual)": "#7c3aed"},
                             labels={"precipitation_sum": "Rainfall (mm)", "date": ""})
                fig.add_hline(y=1, line_dash="dot", line_color="#6b7280", annotation_text="1 mm rain-day threshold",
                              annotation_font_size=11)
                fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(style_fig(fig, 280))
        else:
            with card("Rainfall climatology", f"{city}: average daily rainfall by month, 2015–2025"):
                raw = load_raw()
                clim = (raw[raw.city == city].assign(month=lambda d: d.date.dt.month)
                        .groupby("month")["precipitation_sum"].mean().reset_index())
                clim["selected"] = np.where(clim.month == month, "Selected month", "Other months")
                fig = px.bar(clim, x="month", y="precipitation_sum", color="selected",
                             color_discrete_map={"Selected month": "#0f4c81", "Other months": "#9fb8d6"},
                             labels={"precipitation_sum": "Avg mm / day", "month": "Month"})
                fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
                fig.update_xaxes(dtick=1)
                st.plotly_chart(style_fig(fig, 280))
    with c2:
        with card("Today's readings", "Inputs passed to the model"):
            r = x.iloc[0]
            items = [("Max / min temp", f"{r.temperature_2m_max:.1f} / {r.temperature_2m_min:.1f} °C"),
                     ("Rain today", f"{r.precipitation_sum:.1f} mm"),
                     ("Rain hours", f"{r.precipitation_hours:.0f} h"),
                     ("Rain last 7 days", f"{r.rain_7d_sum:.0f} mm"),
                     ("Humidity", f"{r.relative_humidity_2m_mean:.0f} %"),
                     ("Cloud cover", f"{r.cloud_cover_mean:.0f} %"),
                     ("Pressure", f"{r.pressure_msl_mean:.1f} hPa"),
                     ("Pressure change", f"{r.pressure_change:+.1f} hPa"),
                     ("Max wind", f"{r.wind_speed_10m_max:.0f} km/h"),
                     ("Dew point", f"{r.dew_point_2m_mean:.1f} °C"),
                     ("Radiation", f"{r.shortwave_radiation_sum:.1f} MJ/m²"),
                     ("Season", SEASONS[month].replace("Inter-monsoon", "Inter-m."))]
            st.markdown("<div class='readings'>" + "".join(
                f"<div class='reading'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in items)
                + "</div>", unsafe_allow_html=True)

# ================================================================== PERFORMANCE
elif page == "Model performance":
    page_header("Model performance",
                f"Evaluated on {meta['n_test']:,} unseen city-days from 2024–2025 | {len(metrics)} models")
    fam_best = metrics.loc[metrics.groupby("family")["roc_auc"].idxmax()].set_index("family")
    kpis([(f"Best {f}", f"{fam_best.loc[f, 'roc_auc']:.3f}", f"{fam_best.loc[f, 'model']} (ROC-AUC)",
           FAMILY_COLORS[f]) for f in ["Hybrid", "Bagging", "Boosting", "Stacking"]])

    with st.sidebar:
        st.markdown("<div class='side-label'>Options</div>", unsafe_allow_html=True)
        metric = st.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
        cm_model = st.selectbox("Confusion matrix model", model_names)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card(f"All models: {METRIC_LABELS[metric]}", "Coloured by ensemble family"):
            s = metrics.sort_values(metric)
            fig = px.bar(s, x=metric, y="model", color="family", orientation="h",
                         color_discrete_map=FAMILY_COLORS, labels={metric: METRIC_LABELS[metric], "model": ""},
                         text=s[metric].map("{:.3f}".format))
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_xaxes(range=[s[metric].min() - 0.03, min(1, s[metric].max() + 0.02)])
            fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0))
            st.plotly_chart(style_fig(fig, 420))
    with c2:
        with card("Bagging vs Boosting vs Stacking vs Hybrid", "Best model of each family across all metrics"):
            fam = metrics.loc[metrics.groupby("family")["roc_auc"].idxmax()]
            fam_long = fam.melt(id_vars=["family", "model"], value_vars=list(METRIC_LABELS))
            fam_long["variable"] = fam_long.variable.map(METRIC_LABELS)
            fig = px.bar(fam_long, x="variable", y="value", color="family", barmode="group",
                         color_discrete_map=FAMILY_COLORS, hover_data=["model"],
                         labels={"variable": "", "value": "score"})
            fig.update_yaxes(range=[0.6, 0.92])
            fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0))
            st.plotly_chart(style_fig(fig, 420))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("ROC curves", "Best model per family"):
            fig = go.Figure()
            for _, rw in fam.sort_values("roc_auc", ascending=False).iterrows():
                rc = meta["roc"][rw.model]
                fig.add_trace(go.Scatter(x=rc["fpr"], y=rc["tpr"], mode="lines",
                                         line=dict(color=FAMILY_COLORS[rw.family], width=2),
                                         name=f"{rw.model} ({rw.roc_auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9ca3af"),
                                     showlegend=False))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                              legend=dict(x=0.45, y=0.08, bgcolor="rgba(255,255,255,0.8)"))
            st.plotly_chart(style_fig(fig, 380))
    with c2:
        with card("Confusion matrix", f"{cm_model} | threshold 0.5"):
            cm = np.array(meta["confusion"][cm_model])
            fig = px.imshow(cm, text_auto=",", color_continuous_scale=["#f1f5f9", "#0f4c81"],
                            x=["Predicted: no rain", "Predicted: rain"], y=["Actual: no rain", "Actual: rain"])
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textfont_size=16)
            st.plotly_chart(style_fig(fig, 380))

    with card("Full results table", f"Random Forest out-of-bag accuracy: {meta['rf_oob']:.4f}"):
        show = metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        show = show.rename(columns={**METRIC_LABELS, "family": "Family", "model": "Model",
                                    "train_time_s": "Train time (s)"})
        st.dataframe(show.style.format({v: "{:.4f}" for v in METRIC_LABELS.values()} | {"Train time (s)": "{:.1f}"})
                     .background_gradient(subset=["Accuracy", "F1-score", "ROC-AUC"], cmap="Blues"),
                     hide_index=True, width="stretch")

    with card(f"Best model by climate zone: {meta['best_model']}"):
        z = pd.DataFrame(meta["zone_metrics"]).rename(columns={"climate_zone": "Climate zone", **METRIC_LABELS})
        st.dataframe(z, hide_index=True, width="stretch")

# ================================================================== IMPORTANCE
elif page == "Feature importance":
    page_header("Feature importance", "Which weather signals drive the next-day rain forecast")
    with st.sidebar:
        st.markdown("<div class='side-label'>Options</div>", unsafe_allow_html=True)
        imp_name = st.selectbox("Model", list(meta["importances"].keys()),
                                format_func=lambda n: n.replace("permutation_", "Permutation: "))
        top_n = st.slider("Number of features", 5, 20, 15)
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with card(f"Top {top_n} features", imp_name.replace("permutation_", "Permutation importance: ")):
            imp = pd.Series(meta["importances"][imp_name]).sort_values(ascending=False).head(top_n)[::-1]
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "importance", "y": ""})
            fig.update_traces(marker_color="#0f4c81")
            st.plotly_chart(style_fig(fig, 30 * top_n + 60))
    with c2:
        with card("How to read this"):
            st.markdown("""
**Tree importance** is the mean decrease in impurity, normalised to sum to 1.
**Permutation importance** is the drop in ROC-AUC when a feature is randomly shuffled, which works for any model.

**Main drivers across models**
- **Season** (day-of-year and month): the monsoon calendar
- **Persistence**: today's rain hours and amount, plus the 3- and 7-day totals
- **Atmosphere**: pressure change, humidity, dew point, cloud cover and solar radiation
""")

# ================================================================== DATA
elif page == "Data explorer":
    raw = load_raw()
    page_header("Data explorer", "Daily ERA5 weather for 15 Sri Lankan cities, 2015–2025 (Open-Meteo)")
    summ = raw.groupby(["city", "climate_zone", "province"]).agg(
        annual_rain_mm=("precipitation_sum", lambda s: s.sum() / 11),
        rain_day_share=("precipitation_sum", lambda s: (s >= 1).mean()),
        avg_max_temp=("temperature_2m_max", "mean"),
        lat=("latitude", "first"), lon=("longitude", "first")).reset_index()
    wettest, driest = summ.loc[summ.annual_rain_mm.idxmax()], summ.loc[summ.annual_rain_mm.idxmin()]
    kpis([("Records", f"{len(raw):,}", "daily city observations", "#0f4c81"),
          ("Cities", f"{raw.city.nunique()}", "all 9 provinces", "#0f766e"),
          ("Wettest city", wettest.city, f"{wettest.annual_rain_mm:,.0f} mm / year", "#1e3a8a"),
          ("Driest city", driest.city, f"{driest.annual_rain_mm:,.0f} mm / year", "#d97706")])

    with st.sidebar:
        st.markdown("<div class='side-label'>Options</div>", unsafe_allow_html=True)
        pick = st.multiselect("Cities to compare", cities, default=["Colombo", "Ratnapura", "Trincomalee", "Jaffna"])

    c1, c2 = st.columns([1, 1.3], gap="medium")
    with c1:
        with card("Average annual rainfall", "Bubble size = mm per year"):
            fig = px.scatter_map(summ, lat="lat", lon="lon", size="annual_rain_mm", color="climate_zone",
                                 hover_name="city", hover_data={"annual_rain_mm": ":.0f", "rain_day_share": ":.0%",
                                                                "lat": False, "lon": False},
                                 zoom=6.1, map_style="carto-positron", size_max=26,
                                 color_discrete_map={"Wet": "#0f4c81", "Intermediate": "#0f766e", "Dry": "#d97706"})
            fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0))
            st.plotly_chart(style_fig(fig, 520))
    with c2:
        with card("Monthly rainfall pattern", "Average mm per day; shaded bands show the two monsoons"):
            mon = (raw[raw.city.isin(pick)].assign(month=lambda d: d.date.dt.month)
                   .groupby(["city", "month"])["precipitation_sum"].mean().reset_index())
            fig = px.line(mon, x="month", y="precipitation_sum", color="city", markers=True,
                          labels={"precipitation_sum": "Avg mm / day", "month": "Month"})
            fig.add_vrect(x0=4.5, x1=9.5, fillcolor="#0f766e", opacity=0.07, line_width=0,
                          annotation_text="SW monsoon", annotation_position="top left")
            fig.add_vrect(x0=0.5, x1=2.5, fillcolor="#d97706", opacity=0.07, line_width=0,
                          annotation_text="NE monsoon", annotation_position="top left")
            fig.update_xaxes(dtick=1)
            fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0))
            st.plotly_chart(style_fig(fig, 520))

    with card("City summary"):
        t = summ.drop(columns=["lat", "lon"]).sort_values("annual_rain_mm", ascending=False).rename(columns={
            "city": "City", "climate_zone": "Climate zone", "province": "Province",
            "annual_rain_mm": "Annual rain (mm)", "rain_day_share": "Rain days", "avg_max_temp": "Avg max temp (°C)"})
        st.dataframe(t.style.format({"Annual rain (mm)": "{:,.0f}", "Rain days": "{:.0%}",
                                     "Avg max temp (°C)": "{:.1f}"}), hide_index=True, width="stretch")

# ================================================================== ABOUT
else:
    page_header("About this project", "Lisorthman.V | 225063F")
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        with card("Project overview"):
            st.markdown(f"""
**Task:** binary classification. Will tomorrow be a rain day (at least 1 mm) at a Sri Lankan city?

**Data:** Open-Meteo Historical Weather API (ERA5 reanalysis), 15 cities across all 9 provinces and the
Wet, Intermediate and Dry zones, daily from 2015 to 2025 ({meta['n_train'] + meta['n_test']:,} modelling rows).

**Split:** chronological. Train {meta['train_period']}, test {meta['test_period']}, so there is no look-ahead leakage.

**Features:** today's weather, lagged and rolling rainfall, pressure and humidity change, cyclical season
encodings, monsoon season, and one-hot city, province and climate zone.

**Models**
- **Bagging:** Random Forest, Extra Trees, Bagged Decision Trees
- **Boosting:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Stacking:** RF + LightGBM + KNN + Logistic Regression, with a Logistic Regression meta-learner (5-fold out-of-fold)
- **Hybrid:** Random Forest + ANN (MLP), soft voting
- **Baselines:** Logistic Regression, Decision Tree
""")
    with c2:
        with card("Generative AI disclosure"):
            st.markdown("Claude (Anthropic), via Claude Code, assisted in writing the data download script, the "
                        "modelling notebook and this Streamlit dashboard. All code was reviewed and run by the "
                        "student, and the interpretation of results is the student's own.")
        with card("Source"):
            st.markdown("Weather data: [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), "
                        "ERA5 by ECMWF / Copernicus (CC BY 4.0).\n\n"
                        "Code: [github.com/lisorthman/sri-lanka-rain-ensemble]"
                        "(https://github.com/lisorthman/sri-lanka-rain-ensemble)")

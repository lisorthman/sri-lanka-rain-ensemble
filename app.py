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
st.set_page_config(page_title="Sri Lanka Rain Forecast · Ensembles", page_icon="🌧️", layout="wide")

FAMILY_COLORS = {"Baseline": "#9e9e9e", "Bagging": "#2a9d8f", "Boosting": "#e76f51",
                 "Stacking": "#264653", "Hybrid": "#8e44ad"}
SEASONS = {12: "NE Monsoon", 1: "NE Monsoon", 2: "NE Monsoon", 3: "First Inter-monsoon",
           4: "First Inter-monsoon", 5: "SW Monsoon", 6: "SW Monsoon", 7: "SW Monsoon",
           8: "SW Monsoon", 9: "SW Monsoon", 10: "Second Inter-monsoon", 11: "Second Inter-monsoon"}


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

# ------------------------------------------------------------------ header
st.title("🌧️ Will it rain tomorrow in Sri Lanka?")
st.caption("Ensemble learning dashboard: **Bagging vs Boosting vs Stacking (+ Hybrid RF + ANN)**, "
           "trained on 2015–2023 ERA5 daily weather for 15 Sri Lankan cities (Open-Meteo), "
           "tested on 2024–2025.")

best = metrics.sort_values("roc_auc", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", best.model)
c2.metric("Best ROC-AUC", f"{best.roc_auc:.3f}")
c3.metric("Best accuracy", f"{metrics.accuracy.max():.1%}")
c4.metric("Train / test days", f"{meta['n_train']:,} / {meta['n_test']:,}")

tab_pred, tab_cmp, tab_imp, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model comparison", "🧠 Feature importance", "🗺️ Data explorer", "ℹ️ About"])

# ------------------------------------------------------------------ predict
with tab_pred:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("1 · Inputs")
        model_choice = st.selectbox("Ensemble model", model_names,
                                    format_func=lambda m: f"{m}  ·  {registry[m]['family']}")
        city = st.selectbox("City", sorted(city_info.index), index=sorted(city_info.index).index("Colombo"))
        mode = st.radio("Input mode", ["Replay a real day (2024–2025)", "Custom weather"], horizontal=True)

        city_days = days[days.city == city].sort_values("date")
        if mode.startswith("Replay"):
            d = st.date_input("Date (today)", value=city_days.date.iloc[-200].date(),
                              min_value=city_days.date.min().date(), max_value=city_days.date.max().date())
            row = city_days[city_days.date == pd.Timestamp(d)]
            if row.empty:
                st.warning("No record for that date; showing nearest day.")
                row = city_days.iloc[[np.argmin(np.abs(city_days.date - pd.Timestamp(d)))]]
            x = row[meta["features"]].copy()
            truth = int(row["rain_tomorrow"].iloc[0])
            with st.expander("Weather recorded on this day", expanded=False):
                show = row[["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                            "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
                            "wind_speed_10m_max", "rain_7d_sum"]].T
                show.columns = ["value"]
                st.dataframe(show, width="stretch")
        else:
            truth = None
            base = city_days[meta["features"]].median(numeric_only=True)
            month = st.slider("Month", 1, 12, 6)
            a, b = st.columns(2)
            tmax = a.slider("Max temp (°C)", 10.0, 40.0, float(round(base.temperature_2m_max, 1)))
            tmin = b.slider("Min temp (°C)", 5.0, 30.0, float(round(base.temperature_2m_min, 1)))
            rain = a.slider("Rain today (mm)", 0.0, 150.0, 2.0)
            rain_y = b.slider("Rain yesterday (mm)", 0.0, 150.0, 2.0)
            rain7 = a.slider("Rain last 7 days (mm)", 0.0, 400.0, 20.0)
            hum = b.slider("Humidity (%)", 40.0, 100.0, float(round(base.relative_humidity_2m_mean)))
            pres = a.slider("Sea-level pressure (hPa)", 995.0, 1020.0, float(round(base.pressure_msl_mean, 1)))
            dp = b.slider("Pressure change vs yesterday (hPa)", -6.0, 6.0, 0.0)
            cloud = a.slider("Cloud cover (%)", 0.0, 100.0, float(round(base.cloud_cover_mean)))
            wind = b.slider("Max wind speed (km/h)", 0.0, 60.0, float(round(base.wind_speed_10m_max)))
            wdir = a.slider("Dominant wind direction (°)", 0, 359, 240)
            rhours = b.slider("Hours of rain today", 0.0, 24.0, min(24.0, rain / 1.5))

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

    with right:
        st.subheader("2 · Prediction")
        p = float(load_model(registry[model_choice]["file"]).predict_proba(x)[0, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": f"Chance of rain tomorrow in {city}<br><sub>{model_choice}</sub>"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"},
                   "steps": [{"range": [0, 50], "color": "#fdf0d5"}, {"range": [50, 100], "color": "#d6eaf8"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        gauge.update_layout(height=290, margin=dict(t=70, b=10, l=30, r=30))
        st.plotly_chart(gauge)
        verdict = "🌧️ **Rain expected**" if p >= 0.5 else "☀️ **Likely dry**"
        if truth is None:
            st.markdown(verdict)
        else:
            ok = (p >= 0.5) == bool(truth)
            st.markdown(f"{verdict} &nbsp;·&nbsp; Actual next day: **{'Rain' if truth else 'No rain'}** "
                        f"{'✅ correct' if ok else '❌ missed'}")

        st.markdown("**What does every model say?**")
        all_p = pd.DataFrame([{"model": m, "family": registry[m]["family"],
                               "p_rain": float(load_model(registry[m]["file"]).predict_proba(x)[0, 1])}
                              for m in model_names]).sort_values("p_rain")
        fig = px.bar(all_p, x="p_rain", y="model", color="family", orientation="h",
                     color_discrete_map=FAMILY_COLORS, range_x=[0, 1],
                     labels={"p_rain": "P(rain tomorrow)", "model": ""})
        fig.add_vline(x=0.5, line_dash="dash", line_color="black")
        fig.update_layout(height=430, margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig)

# ------------------------------------------------------------------ comparison
with tab_cmp:
    st.subheader("Test-set performance (2024–2025, unseen years)")
    show = metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    st.dataframe(show.style.format({c: "{:.4f}" for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                                   | {"train_time_s": "{:.1f}"})
                 .background_gradient(subset=["accuracy", "f1", "roc_auc"], cmap="Greens"),
                 width="stretch")

    metric = st.radio("Metric", ["roc_auc", "accuracy", "f1", "recall", "precision"], horizontal=True)
    c1, c2 = st.columns(2)
    s = show.sort_values(metric)
    fig = px.bar(s, x=metric, y="model", color="family", orientation="h", color_discrete_map=FAMILY_COLORS,
                 title=f"All models: {metric}")
    fig.update_xaxes(range=[s[metric].min() - 0.03, min(1, s[metric].max() + 0.01)])
    c1.plotly_chart(fig)

    fam = show.loc[show.groupby("family")[metric].idxmax()]
    fam_long = fam.melt(id_vars=["family", "model"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    fig = px.bar(fam_long, x="variable", y="value", color="family", barmode="group",
                 color_discrete_map=FAMILY_COLORS, hover_data=["model"],
                 title="Bagging vs Boosting vs Stacking vs Hybrid (best of each family)")
    fig.update_yaxes(range=[0.6, 1])
    c2.plotly_chart(fig)

    c1, c2 = st.columns(2)
    sel = c1.multiselect("ROC curves for", model_names, default=list(fam.model))
    fig = go.Figure()
    for m in sel:
        r = meta["roc"][m]
        auc = metrics.set_index("model").loc[m, "roc_auc"]
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{m} ({auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                             showlegend=False))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=450,
                      title="ROC curves")
    c1.plotly_chart(fig)

    cm_model = c2.selectbox("Confusion matrix for", model_names)
    cm = np.array(meta["confusion"][cm_model])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: no rain", "Pred: rain"], y=["Actual: no rain", "Actual: rain"],
                    title=f"Confusion matrix: {cm_model}")
    fig.update_layout(height=450, coloraxis_showscale=False)
    c2.plotly_chart(fig)

    st.markdown(f"**Random Forest Out-of-Bag accuracy:** {meta['rf_oob']:.4f}. Bagging gives a free "
                "validation estimate from the rows each tree never saw.")
    st.markdown(f"**Best model ({meta['best_model']}) by climate zone**")
    st.dataframe(pd.DataFrame(meta["zone_metrics"]).set_index("climate_zone"), width="stretch")

# ------------------------------------------------------------------ importance
with tab_imp:
    st.subheader("Which weather signals drive the forecast?")
    imp_name = st.selectbox("Model", list(meta["importances"].keys()))
    imp = pd.Series(meta["importances"][imp_name]).sort_values()
    fig = px.bar(imp, orientation="h", labels={"value": "importance", "index": ""},
                 title=f"Top features: {imp_name}")
    fig.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig)
    st.info("Tree importances = mean impurity decrease (normalised). "
            "Permutation importance = drop in ROC-AUC when a feature is shuffled (model-agnostic).")

# ------------------------------------------------------------------ data
with tab_data:
    raw = load_raw()
    st.subheader("Sri Lankan daily weather 2015–2025 (15 cities)")
    m = raw.assign(month=raw.date.dt.month).groupby(["city", "climate_zone"]).agg(
        annual_rain_mm=("precipitation_sum", lambda s: s.sum() / 11),
        rain_day_share=("precipitation_sum", lambda s: (s >= 1).mean()),
        lat=("latitude", "first"), lon=("longitude", "first")).reset_index()
    c1, c2 = st.columns([1, 1.2])
    fig = px.scatter_map(m, lat="lat", lon="lon", size="annual_rain_mm", color="climate_zone",
                         hover_name="city", hover_data={"annual_rain_mm": ":.0f", "rain_day_share": ":.0%"},
                         zoom=6.2, height=560, map_style="carto-positron", size_max=28,
                         title="Average annual rainfall (bubble size)")
    c1.plotly_chart(fig)
    pick = c2.multiselect("Cities", sorted(raw.city.unique()), default=["Colombo", "Ratnapura", "Trincomalee", "Jaffna"])
    mon = (raw[raw.city.isin(pick)].assign(month=raw.date.dt.month)
           .groupby(["city", "month"])["precipitation_sum"].mean().reset_index())
    fig = px.line(mon, x="month", y="precipitation_sum", color="city", markers=True,
                  labels={"precipitation_sum": "avg mm / day"}, title="Monthly rainfall pattern (monsoons)")
    fig.add_vrect(x0=4.5, x1=9.5, fillcolor="teal", opacity=0.07, annotation_text="SW monsoon")
    fig.add_vrect(x0=0.5, x1=2.5, fillcolor="orange", opacity=0.07, annotation_text="NE monsoon")
    c2.plotly_chart(fig)
    with st.expander("Raw data sample"):
        st.dataframe(raw.sample(200, random_state=1).sort_values(["city", "date"]), width="stretch")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
### Project
Binary classification: **will tomorrow be a rain day (≥ 1 mm) at a Sri Lankan city?**

* **Data:** Open-Meteo Historical Weather API (ERA5 reanalysis), 15 cities across all 9 provinces and the Wet,
  Intermediate and Dry zones, daily 2015–2025.
* **Split:** chronological. Train {meta['train_period']}, test {meta['test_period']}, so no look-ahead leakage.
* **Features:** today's weather, lagged/rolling rainfall, pressure & humidity change, cyclical season,
  monsoon season, city / province / climate zone (one-hot).
* **Models:** Bagging (Random Forest, Extra Trees, Bagged Trees), Boosting (AdaBoost, Gradient Boosting, XGBoost,
  LightGBM, CatBoost), Stacking (RF + LightGBM + KNN + LR → LR meta-learner, 5-fold out-of-fold),
  Hybrid (Random Forest + ANN soft voting), plus Logistic Regression & Decision Tree baselines.

### Generative AI disclosure
Claude (Anthropic), via Claude Code, assisted in writing the data download script, the modelling notebook
and this Streamlit dashboard. All code was reviewed and run by the student, and the interpretation of
results is the student's own.
""")

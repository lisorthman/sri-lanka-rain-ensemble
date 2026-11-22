# 🌧️ Sri Lanka Rain Forecast: Ensemble Learning Dashboard

Predict whether **tomorrow will be a rain day (≥ 1 mm)** at 15 Sri Lankan cities, and compare
**Bagging vs Boosting vs Stacking** (plus a **Hybrid Random Forest + ANN**) in an interactive Streamlit dashboard.

## Dataset (Sri Lankan, real)
* Daily weather for **15 cities** (all 9 provinces; Wet, Intermediate and Dry zones), **2015–2025**, 60,270 rows
* Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), ERA5 reanalysis (CC BY 4.0)
* Re-download any time with `python fetch_data.py`

## Results (test set: 2024–2025, unseen years)

| Family | Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Hybrid | RF + ANN | 0.774 | 0.825 | 0.824 | **0.844** |
| Bagging | Extra Trees | 0.764 | 0.776 | 0.808 | 0.841 |
| Bagging | Random Forest (OOB 0.795) | 0.771 | 0.811 | 0.819 | 0.841 |
| Boosting | Gradient Boosting | 0.772 | 0.840 | 0.825 | 0.841 |
| Boosting | XGBoost | 0.767 | 0.779 | 0.811 | 0.840 |
| Boosting | CatBoost | 0.761 | 0.768 | 0.805 | 0.840 |
| Stacking | RF + LGBM + KNN + LR → LR | 0.774 | **0.879** | **0.833** | 0.839 |
| Bagging | Bagged Decision Trees | 0.769 | 0.805 | 0.817 | 0.838 |
| Boosting | LightGBM | 0.764 | 0.779 | 0.808 | 0.838 |
| Baseline | Logistic Regression | 0.755 | 0.757 | 0.798 | 0.831 |
| Boosting | AdaBoost | 0.756 | 0.775 | 0.802 | 0.826 |
| Baseline | Decision Tree | 0.733 | 0.736 | 0.779 | 0.807 |

See section 8 of the notebook for the full interpretation.

## Project structure
```
ensemble_rain_sri_lanka.ipynb   ← main notebook (EDA, preprocessing, 12 models, evaluation, interpretation)
app.py                          ← Streamlit dashboard
fetch_data.py                   ← downloads the dataset from Open-Meteo
data/sri_lanka_weather.csv      ← raw dataset
models/                         ← trained pipelines (.joblib), metrics & test days used by the dashboard
requirements.txt
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: XGBoost/LightGBM need OpenMP →  brew install libomp
jupyter notebook ensemble_rain_sri_lanka.ipynb   # optional: retrains & re-saves models (~6 min)
streamlit run app.py
```

## Generative AI disclosure
Claude (Anthropic), via Claude Code, helped write the data download script, the notebook code and the
Streamlit dashboard, and drafted parts of the explanations. All code was reviewed and run by the student,
and the interpretation is the student's own.

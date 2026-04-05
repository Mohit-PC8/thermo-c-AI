# ThermoCardial AI — Setup Guide

## Project Structure

```
your_project/
├── app.py                ← Streamlit frontend (this file)
├── requirements.txt      ← Dependencies
├── model.pkl             ← Trained GradientBoosting model
├── scaler.pkl            ← Final feature scaler
├── scaler_gmm.pkl        ← GMM feature scaler
├── gmm.pkl               ← Gaussian Mixture Model
├── columns.pkl           ← Final column list (30 features)
└── thermocardial.db      ← SQLite DB (auto-created on first run)
```

## Step 1 — Export .pkl files from Google Colab

Run **Block 6** in your Colab notebook. It saves:
- `model.pkl`
- `scaler.pkl`
- `scaler_gmm.pkl`
- `gmm.pkl`
- `columns.pkl`

Download all five files from Colab (Files panel → right-click → Download).

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3 — Run the app

Place `app.py` and all `.pkl` files in the **same folder**, then:

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Features

| Section | Description |
|---|---|
| **Patient Form** | 13 clinical fields with dropdowns & validated number inputs |
| **Prediction Result** | Risk label, probability, severity score (0–10) |
| **Gauge Chart** | Visual risk meter |
| **Radar Chart** | 8-axis patient parameter profile |
| **Bar Chart** | Clinical parameter magnitudes |
| **Community Stats** | Cumulative positive/negative counts, donut, trend line, histogram |
| **SQLite DB** | Every submission stored with timestamp & all 13 inputs |

## Database Schema

Table: `predictions`

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | Submission datetime |
| age…thal | REAL | 13 clinical inputs |
| prediction | INTEGER | 0=Low Risk, 1=High Risk |
| probability | REAL | Model probability score |
| severity | REAL | probability × 10 (0–10 scale) |

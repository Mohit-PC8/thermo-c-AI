"""
ThermoCardial AI — Heart Disease Prediction System
Streamlit Frontend + SQLite + PDF Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os, io
from datetime import datetime
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ThermoCardial AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (label visibility fixed)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary:#0a0514; --bg-card:#130d24; --bg-input:#1a1130;
    --accent-violet:#7c3aed; --accent-indigo:#4f46e5;
    --accent-pink:#ec4899; --accent-cyan:#06b6d4;
    --accent-emerald:#10b981; --accent-red:#ef4444;
    --text-primary:#f1e8ff; --text-muted:#b8aace;
    --border:#2e1f4a; --glow:rgba(124,58,237,0.35);
}
html,body,[class*="css"]{
    font-family:'Space Grotesk',sans-serif!important;
    font-size:18px !important;
    background-color:var(--bg-primary)!important;
    color:var(--text-primary)!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 4rem 2rem!important;max-width:1600px!important;}

/* ── Hero ── */
.hero{text-align:center;padding:4rem 2rem 3rem;
  background:linear-gradient(135deg,rgba(15,23,42,0.95) 0%,rgba(30,41,59,0.95) 50%,rgba(15,23,42,0.95) 100%);
  border-radius:20px;
  border:2px solid rgba(124,58,237,0.3);
  box-shadow:0 8px 32px rgba(0,0,0,0.4),0 16px 64px rgba(0,0,0,0.2);
  margin-bottom:3rem;}
.hero-title{font-size:8.5rem;font-weight:900;letter-spacing:-2px;line-height:1.1;margin:0;color:#ffffff;text-shadow:0 2px 4px rgba(0,0,0,0.8),0 4px 8px rgba(0,0,0,0.4);}
.hero-sub{margin-top:1.5rem;font-size:2rem;color:#e2e8f4;letter-spacing:3px;font-weight:600;text-shadow:0 1px 3px rgba(0,0,0,0.6);}

/* ── Section headers ── */
.section-label{font-size:.7rem;letter-spacing:3px;text-transform:uppercase;
  color:var(--accent-violet);font-weight:600;margin-bottom:.4rem;}
.section-title{font-size:2rem;font-weight:700;color:#1e3a8a;margin:0 0 1.2rem;}

/* ── Cards ── */
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;
  padding:1.5rem;margin-bottom:1.2rem;}

/* ── FIX: ALL labels → pure black, max contrast ── */
label,
.stSelectbox label,
.stNumberInput label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] span,
div[data-testid="stWidgetLabel"] > div,
.stForm label,
p[data-testid="stWidgetLabel"],
[data-baseweb="form-control-label"],
[data-baseweb="label"] {
    color: #000000 !important;
    font-size: .88rem !important;
    font-weight: 600 !important;
}
/* Tooltip (?) icon */
div[data-testid="stWidgetLabel"] button svg { color:#7c3aed!important; }

/* ── FIX: white cursor caret in dark input fields ── */
div[data-testid="stNumberInput"] input {
    caret-color: #ffffff !important;
}
div[data-testid="stTextInput"] input {
    caret-color: #ffffff !important;
}

/* Column sub-headings */
.col-heading{
    color:#1e3a8a;font-size:.78rem;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid var(--border);
    padding-bottom:.5rem;margin-bottom:1rem;
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background:var(--bg-input)!important;border:1px solid var(--border)!important;
    border-radius:8px!important;color:var(--text-primary)!important;
    font-family:'Space Grotesk',sans-serif!important;
}
div[data-testid="stNumberInput"] input:focus{
    border-color:var(--accent-violet)!important;
    box-shadow:0 0 0 2px rgba(124,58,237,.3)!important;
}

/* ── Result Boxes ── */
.result-positive{background:linear-gradient(135deg,rgba(239,68,68,.15),rgba(239,68,68,.05));
  border:1px solid rgba(239,68,68,.45);border-radius:16px;padding:1.8rem;text-align:center;}
.result-negative{background:linear-gradient(135deg,rgba(16,185,129,.15),rgba(16,185,129,.05));
  border:1px solid rgba(16,185,129,.45);border-radius:16px;padding:1.8rem;text-align:center;}
.result-icon{font-size:3.5rem;line-height:1;margin-bottom:.4rem;}
.result-label{font-size:1.5rem;font-weight:700;margin-bottom:.3rem;}
.result-prob{font-size:.9rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;}

/* ── Stat Tiles ── */
.stat-tile{border-radius:12px;padding:1.2rem 1.4rem;display:flex;flex-direction:column;gap:.3rem;}
.stat-positive{background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(239,68,68,.06));border:1px solid rgba(239,68,68,.35);}
.stat-negative{background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(16,185,129,.06));border:1px solid rgba(16,185,129,.35);}
.stat-total{background:linear-gradient(135deg,rgba(79,70,229,.18),rgba(79,70,229,.06));border:1px solid rgba(79,70,229,.35);}
.stat-num{font-size:2.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;}
.stat-desc{font-size:.8rem;color:#1e3a8a;text-transform:uppercase;letter-spacing:1px;
    font-weight:600;
}

/* ── Submit / Download Buttons ── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
.stButton > button {
    background:linear-gradient(135deg,var(--accent-violet),var(--accent-indigo))!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    padding:.7rem 2rem!important;font-size:1rem!important;font-weight:600!important;
    width:100%!important;letter-spacing:.5px!important;
    font-family:'Space Grotesk',sans-serif!important;
    transition:opacity .2s,box-shadow .2s!important;
}
.stButton > button:hover{opacity:.88!important;box-shadow:0 0 20px rgba(124,58,237,.5)!important;}
.stDownloadButton > button {
    background:linear-gradient(135deg,#0f766e,#0d9488)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    padding:.65rem 1.5rem!important;font-size:.95rem!important;font-weight:600!important;
    letter-spacing:.5px!important;width:100%!important;
    font-family:'Space Grotesk',sans-serif!important;
}

hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg-primary);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

/* ── Google Anti-Gravity Bubble Cursor Effect ── */
.bubble-cursor {
    position: fixed;
    pointer-events: none;
    z-index: 9999;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.8), rgba(79,70,229,0.6));
    box-shadow: 0 0 10px rgba(124,58,237,0.6), 0 0 20px rgba(124,58,237,0.3);
    transform: translate(-50%, -50%);
    transition: none;
    animation: bubbleFloat 2s ease-in-out infinite;
}

.bubble-trail {
    position: fixed;
    pointer-events: none;
    z-index: 9998;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: opacity 0.8s ease-out, transform 0.8s ease-out;
}

@keyframes bubbleFloat {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.1); }
}

@keyframes ripple {
    0% {
        transform: translate(-50%, -50%) scale(0);
        opacity: 1;
    }
    100% {
        transform: translate(-50%, -50%) scale(3);
        opacity: 0;
    }
}

.ripple-effect {
    position: fixed;
    pointer-events: none;
    z-index: 9997;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 2px solid rgba(124,58,237,0.4);
    transform: translate(-50%, -50%);
    animation: ripple 1s ease-out;
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BUBBLE CURSOR JAVASCRIPT
# ─────────────────────────────────────────────
bubble_js = """
<script>
// Simple bubble cursor for Streamlit compatibility
(function() {
    'use strict';
    
    let cursor = null;
    let trails = [];
    const maxTrails = 6;
    const colors = ['#7c3aed', '#4f46e5', '#ec4899', '#06b6d4', '#10b981', '#ef4444'];
    
    function createCursor() {
        if (cursor) return;
        
        cursor = document.createElement('div');
        cursor.style.cssText = `
            position: fixed;
            pointer-events: none;
            z-index: 9999;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: radial-gradient(circle, #7c3aed, #4f46e5);
            box-shadow: 0 0 15px #7c3aed, 0 0 30px #7c3aed40;
            transform: translate(-50%, -50%);
            transition: all 0.15s ease-out;
        `;
        document.body.appendChild(cursor);
        
        console.log('Bubble cursor created');
    }
    
    function createTrail(x, y) {
        if (trails.length >= maxTrails) {
            const oldTrail = trails.shift();
            if (oldTrail && oldTrail.parentNode) {
                oldTrail.parentNode.removeChild(oldTrail);
            }
        }
        
        const trail = document.createElement('div');
        const size = Math.random() * 8 + 4;
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        trail.style.cssText = `
            position: fixed;
            pointer-events: none;
            z-index: 9998;
            width: ${size}px;
            height: ${size}px;
            border-radius: 50%;
            background: radial-gradient(circle, ${color}, ${color}40);
            box-shadow: 0 0 ${size}px ${color};
            left: ${x}px;
            top: ${y}px;
            transform: translate(-50%, -50%);
            opacity: 0.8;
            transition: opacity 1s ease-out, transform 1s ease-out;
        `;
        
        document.body.appendChild(trail);
        trails.push(trail);
        
        setTimeout(() => {
            trail.style.opacity = '0';
            trail.style.transform = 'translate(-50%, -50%) scale(0.3)';
        }, 50);
        
        setTimeout(() => {
            if (trail.parentNode) {
                trail.parentNode.removeChild(trail);
            }
            const index = trails.indexOf(trail);
            if (index > -1) {
                trails.splice(index, 1);
            }
        }, 1000);
    }
    
    function handleMouseMove(e) {
        if (!cursor) {
            createCursor();
        }
        
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
        
        if (Math.random() > 0.4) {
            createTrail(e.clientX, e.clientY);
        }
    }
    
    function init() {
        document.addEventListener('mousemove', handleMouseMove);
        console.log('Bubble cursor initialized');
    }
    
    // Multiple initialization attempts
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 100);
    }
    
    // Fallback
    setTimeout(init, 1000);
})();
</script>
"""

# Alternative method for JavaScript execution
components.html(bubble_js, height=0)

# ─────────────────────────────────────────────
# CATEGORICAL OPTION MAPS  (label → numeric)
# ─────────────────────────────────────────────
SEX_MAP    = {"0 — Female": 0, "1 — Male": 1}
CP_MAP     = {
    "0 — Typical Angina": 0,
    "1 — Atypical Angina": 1,
    "2 — Non-Anginal Pain": 2,
    "3 — Asymptomatic": 3,
}
FBS_MAP    = {"0 — No  (≤ 120 mg/dl)": 0, "1 — Yes  (> 120 mg/dl)": 1}
RESTECG_MAP= {
    "0 — Normal": 0,
    "1 — ST-T Wave Abnormality": 1,
    "2 — Left Ventricular Hypertrophy": 2,
}
EXANG_MAP  = {"0 — No": 0, "1 — Yes": 1}
SLOPE_MAP  = {"0 — Upsloping": 0, "1 — Flat": 1, "2 — Downsloping": 2}
CA_MAP     = {
    "0 — Zero Vessels": 0, "1 — One Vessel": 1,
    "2 — Two Vessels": 2, "3 — Three Vessels": 3, "4 — Four Vessels": 4,
}
THAL_MAP   = {
    "0 — Normal": 0,
    "1 — Fixed Defect": 1,
    "2 — Reversible Defect (mild)": 2,
    "3 — Reversible Defect (severe)": 3,
}

# ─────────────────────────────────────────────
# FIELD META  (full form, unit, range, tip)
# ─────────────────────────────────────────────
FIELD_META = {
    "age":      {"full": "Age",
                 "unit": "years",
                 "range": "29 – 80 (dataset range)",
                 "tip": "Patient's age in years. Range in dataset: 29–80. Mean ≈ 50 years."},
    "sex":      {"full": "Sex (Biological)",
                 "unit": "categorical",
                 "range": "0 = Female | 1 = Male",
                 "tip": "Biological sex. Males have statistically higher cardiac risk."},
    "cp":       {"full": "Chest Pain Type (CP)",
                 "unit": "categorical (0–3)",
                 "range": "0=Typical Angina | 1=Atypical | 2=Non-Anginal | 3=Asymptomatic",
                 "tip": "Type of chest pain. Asymptomatic (3) is ironically the highest-risk category."},
    "trestbps": {"full": "Trestbps — Resting Blood Pressure",
                 "unit": "mm Hg",
                 "range": "90 – 200 mm Hg. Normal: <120. High: ≥130.",
                 "tip": "Resting blood pressure on admission. Dataset range: 90–200 mm Hg, mean ≈ 133."},
    "chol":     {"full": "Chol — Serum Cholesterol",
                 "unit": "mg/dl",
                 "range": "120 – 564 mg/dl. Desirable: <200. High: ≥240.",
                 "tip": "Total serum cholesterol. Dataset range: 120–564 mg/dl, mean ≈ 237."},
    "fbs":      {"full": "FBS — Fasting Blood Sugar",
                 "unit": "categorical",
                 "range": "0 = ≤120 mg/dl (normal) | 1 = >120 mg/dl (elevated)",
                 "tip": "Whether fasting blood sugar exceeds 120 mg/dl. Elevated FBS may indicate diabetes."},
    "restecg":  {"full": "Restecg — Resting Electrocardiographic Results",
                 "unit": "categorical (0–2)",
                 "range": "0=Normal | 1=ST-T Abnormality | 2=LV Hypertrophy",
                 "tip": "ECG findings at rest. Abnormal readings increase cardiac risk."},
    "thalach":  {"full": "Thalach — Maximum Heart Rate Achieved",
                 "unit": "bpm",
                 "range": "70 – 210 bpm. Dataset mean ≈ 151 bpm.",
                 "tip": "Maximum heart rate during stress test. Dataset range: 70–210 bpm, mean ≈ 151. Target = 220 − Age."},
    "exang":    {"full": "Exang — Exercise Induced Angina",
                 "unit": "categorical",
                 "range": "0 = No | 1 = Yes",
                 "tip": "Whether exercise caused chest pain (angina). Yes = higher cardiac risk."},
    "oldpeak":  {"full": "Oldpeak — ST Depression Induced by Exercise",
                 "unit": "mm (relative to rest)",
                 "range": "0.0 – 6.5 mm. Normal: 0. Concerning: >2.",
                 "tip": "ST-segment depression during exercise vs. rest. Dataset range: 0–6.5, mean ≈ 1.3. Higher = worse."},
    "slope":    {"full": "Slope — Slope of Peak Exercise ST Segment",
                 "unit": "categorical (0–2)",
                 "range": "0=Upsloping (good) | 1=Flat | 2=Downsloping (worst)",
                 "tip": "The slope of the ST segment at peak exercise. Downsloping is most concerning."},
    "ca":       {"full": "CA — Number of Major Vessels (Fluoroscopy)",
                 "unit": "count (0–4)",
                 "range": "0 (no disease) – 4 (severe). Dataset mean ≈ 0.82.",
                 "tip": "Number of major coronary vessels colored by fluoroscopy. More = greater disease burden."},
    "thal":     {"full": "Thal — Thalassemia / Nuclear Stress Test",
                 "unit": "categorical (0–3)",
                 "range": "0=Normal | 1=Fixed Defect | 2=Reversible (mild) | 3=Reversible (severe)",
                 "tip": "Result of thallium nuclear stress test. Reversible defect indicates ischemia."},
}

# Health tips per field
HEALTH_TIPS = {
    "trestbps": {
        "normal": "Your resting blood pressure is in a healthy range. Maintain through regular exercise and a low-sodium diet.",
        "elevated": "Resting BP is elevated. Reduce sodium intake to <2300 mg/day, exercise 150 min/week, limit alcohol.",
        "high": "High resting BP detected. Consult a cardiologist. Consider DASH diet, stress management, and medication review.",
    },
    "chol": {
        "desirable": "Cholesterol is in the desirable range. Maintain with a heart-healthy diet rich in fiber and omega-3.",
        "borderline": "Borderline cholesterol. Reduce saturated fats, increase soluble fiber (oats, beans), exercise regularly.",
        "high": "High cholesterol. Consult a physician about statins. Avoid trans fats, red meat; increase plant sterols.",
    },
    "thalach": {
        "good": "Maximum heart rate is within expected range for your age. Cardiovascular fitness appears adequate.",
        "low": "Lower-than-expected max HR may indicate reduced cardiac reserve. Consult a cardiologist for exercise stress evaluation.",
    },
    "oldpeak": {
        "normal": "No significant ST depression. This is a favorable cardiac indicator.",
        "moderate": "Moderate ST depression observed. Lifestyle modification and cardiology follow-up recommended.",
        "high": "Significant ST depression. This is a strong predictor of coronary artery disease. Immediate cardiology evaluation advised.",
    },
    "fbs": {
        "normal": "Fasting blood sugar is normal. Maintain through a balanced low-glycaemic diet.",
        "elevated": "Elevated fasting blood sugar indicates possible pre-diabetes or diabetes. Consult an endocrinologist and monitor HbA1c.",
    },
    "ca": {
        "none": "No major vessels affected by disease — a positive finding.",
        "some": "One or more vessels affected. Medication adherence, lifestyle changes, and regular cardiology review are essential.",
    },
    "exang": {
        "no": "No exercise-induced angina — favorable.",
        "yes": "Exercise-induced angina present. Avoid strenuous exertion until evaluated. Cardiology stress test recommended.",
    },
    "thal": {
        "normal": "Normal thallium stress test result.",
        "fixed": "Fixed defect indicates prior myocardial infarction (heart attack). Cardiac rehabilitation and medication review are important.",
        "reversible": "Reversible defect indicates ischemia (reduced blood flow during stress). Revascularization may be indicated.",
    },
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = "thermocardial.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL,
            fbs REAL, restecg REAL, thalach REAL, exang REAL,
            oldpeak REAL, slope REAL, ca REAL, thal REAL,
            prediction INTEGER, probability REAL, severity REAL
        )
    """)
    conn.commit(); conn.close()

def save_prediction(inputs, pred, prob, sev):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (timestamp,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,prediction,probability,severity)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          inputs["age"],inputs["sex"],inputs["cp"],inputs["trestbps"],inputs["chol"],
          inputs["fbs"],inputs["restecg"],inputs["thalach"],inputs["exang"],
          inputs["oldpeak"],inputs["slope"],inputs["ca"],inputs["thal"],
          pred, prob, sev))
    conn.commit(); conn.close()

def load_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close(); return df

# ─────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────
MODEL_FILES = ["model.pkl","scaler.pkl","scaler_gmm.pkl","gmm.pkl","columns.pkl"]
ORIG_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
             'thalach','exang','oldpeak','slope','ca','thal']

@st.cache_resource
def load_predictor():
    missing = [f for f in MODEL_FILES if not os.path.exists(f)]
    if missing:
        return None, missing
    return dict(
        model=joblib.load("model.pkl"), scaler=joblib.load("scaler.pkl"),
        scaler_gmm=joblib.load("scaler_gmm.pkl"), gmm=joblib.load("gmm.pkl"),
        columns=joblib.load("columns.pkl")
    ), []

def safe_div(a, b): return a / (b + 1e-6)

def add_features(df):
    df = df.copy()
    df['heart_rate_reserve']       = df['thalach'] - (220 - df['age'])
    df['st_depression_ratio']      = safe_div(df['oldpeak'], df['thalach'])
    df['rate_pressure_product']    = (df['trestbps'] * df['thalach']) / 1000
    df['bp_heart_rate_ratio']      = safe_div(df['trestbps'], df['thalach'])
    df['age_normalized_heart_rate']= safe_div(df['thalach'], (220 - df['age']))
    df['exercise_induced_stress']  = df['exang'] * df['oldpeak']
    df['cholesterol_age_ratio']    = safe_div(df['chol'], df['age'])
    df['fbs_stress']               = df['fbs'] * df['oldpeak']
    df['cardiac_work']             = df['thalach'] * (df['trestbps'] - 80)
    df['oxygen_demand_index']      = safe_div(df['thalach'] * df['oldpeak'], (220 - df['age']))
    df['metabolic_equivalent']     = df['thalach'] / 100
    df['thermodynamic_strain']     = safe_div(df['oldpeak'] * df['trestbps'], df['thalach'])
    return df

def run_prediction(artifacts, patient_vals):
    df = pd.DataFrame([patient_vals], columns=ORIG_COLS)
    df = add_features(df)
    gmm_scaled = artifacts["scaler_gmm"].transform(df)
    label = artifacts["gmm"].predict(gmm_scaled)
    pheno = pd.get_dummies(label, prefix='phenotype')
    for i in range(5):
        col = f'phenotype_{i}'
        if col not in pheno.columns: pheno[col] = 0
    df = pd.concat([df, pheno], axis=1)
    df = df.reindex(columns=artifacts["columns"], fill_value=0)
    df_scaled = artifacts["scaler"].transform(df)
    pred = artifacts["model"].predict(df_scaled)[0]
    prob = artifacts["model"].predict_proba(df_scaled)[0][1]
    return {"prediction": int(pred), "probability": float(prob), "severity": float(prob * 10)}

# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def build_health_recommendations(inputs, pred, prob):
    """Return a list of (field_full_name, tip_text) tuples."""
    recs = []
    age = inputs["age"]

    # Blood pressure
    bp = inputs["trestbps"]
    if bp < 120:
        recs.append(("Resting Blood Pressure (Trestbps)", HEALTH_TIPS["trestbps"]["normal"]))
    elif bp < 130:
        recs.append(("Resting Blood Pressure (Trestbps)", HEALTH_TIPS["trestbps"]["elevated"]))
    else:
        recs.append(("Resting Blood Pressure (Trestbps)", HEALTH_TIPS["trestbps"]["high"]))

    # Cholesterol
    chol = inputs["chol"]
    if chol < 200:
        recs.append(("Serum Cholesterol (Chol)", HEALTH_TIPS["chol"]["desirable"]))
    elif chol < 240:
        recs.append(("Serum Cholesterol (Chol)", HEALTH_TIPS["chol"]["borderline"]))
    else:
        recs.append(("Serum Cholesterol (Chol)", HEALTH_TIPS["chol"]["high"]))

    # Max heart rate
    expected_max_hr = 220 - age
    thalach = inputs["thalach"]
    if thalach >= 0.85 * expected_max_hr:
        recs.append(("Maximum Heart Rate (Thalach)", HEALTH_TIPS["thalach"]["good"]))
    else:
        recs.append(("Maximum Heart Rate (Thalach)", HEALTH_TIPS["thalach"]["low"]))

    # ST Depression
    oldpeak = inputs["oldpeak"]
    if oldpeak <= 1.0:
        recs.append(("ST Depression (Oldpeak)", HEALTH_TIPS["oldpeak"]["normal"]))
    elif oldpeak <= 2.0:
        recs.append(("ST Depression (Oldpeak)", HEALTH_TIPS["oldpeak"]["moderate"]))
    else:
        recs.append(("ST Depression (Oldpeak)", HEALTH_TIPS["oldpeak"]["high"]))

    # FBS
    if inputs["fbs"] == 0:
        recs.append(("Fasting Blood Sugar (FBS)", HEALTH_TIPS["fbs"]["normal"]))
    else:
        recs.append(("Fasting Blood Sugar (FBS)", HEALTH_TIPS["fbs"]["elevated"]))

    # CA
    if inputs["ca"] == 0:
        recs.append(("Major Vessels (CA)", HEALTH_TIPS["ca"]["none"]))
    else:
        recs.append(("Major Vessels (CA)", HEALTH_TIPS["ca"]["some"]))

    # Exang
    if inputs["exang"] == 0:
        recs.append(("Exercise-Induced Angina (Exang)", HEALTH_TIPS["exang"]["no"]))
    else:
        recs.append(("Exercise-Induced Angina (Exang)", HEALTH_TIPS["exang"]["yes"]))

    # Thal
    t = inputs["thal"]
    if t == 0:
        recs.append(("Thalassemia (Thal)", HEALTH_TIPS["thal"]["normal"]))
    elif t == 1:
        recs.append(("Thalassemia (Thal)", HEALTH_TIPS["thal"]["fixed"]))
    else:
        recs.append(("Thalassemia (Thal)", HEALTH_TIPS["thal"]["reversible"]))

    return recs


def generate_pdf_report(inputs: dict, result: dict) -> bytes:  # WHITE BG VERSION
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=22*mm, rightMargin=22*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            title="ThermoCardial AI — Cardiac Risk Report")

    # ── ALL colors safe on WHITE background PDF ──
    C_WHITE      = colors.white
    C_ROW_ALT    = colors.HexColor("#ede9fe")
    C_VIOLET     = colors.HexColor("#7c3aed")
    C_HDR_BG     = colors.HexColor("#4338ca")
    C_TEAL       = colors.HexColor("#0d9488")
    C_RED        = colors.HexColor("#dc2626")
    C_GREEN      = colors.HexColor("#16a34a")
    C_AMBER      = colors.HexColor("#b45309")
    C_BLACK      = colors.HexColor("#111111")
    C_BORDER     = colors.HexColor("#c4b5fd")
    C_BANNER_HIGH= colors.HexColor("#fee2e2")
    C_BANNER_LOW = colors.HexColor("#d1fae5")
    C_DIS_BG     = colors.HexColor("#f3f0ff")

    pred  = result["prediction"]
    prob  = result["probability"]
    sev   = result["severity"]
    stamp = datetime.now().strftime("%B %d, %Y  %H:%M")
    verdict_color = C_RED   if pred == 1 else C_GREEN
    verdict_label = "HIGH CARDIAC RISK DETECTED" if pred == 1 else "LOW CARDIAC RISK — NORMAL"
    banner_bg     = C_BANNER_HIGH if pred == 1 else C_BANNER_LOW

    base = getSampleStyleSheet()

    # ── FIX: S() accepts an optional parent kwarg (defaults to base["Normal"]) ──
    # This prevents "multiple values for keyword argument 'parent'" when callers
    # pass parent= explicitly.
    def S(name, parent=None, **kw):
        p = parent if parent is not None else base["Normal"]
        return ParagraphStyle(name, parent=p, **kw)

    s_tag     = S("tag",  fontSize=8,  textColor=C_BLACK,  fontName="Helvetica",
                          alignment=TA_CENTER, spaceAfter=2, leading=12)
    s_stamp   = S("stmp", fontSize=8,  textColor=C_BLACK,  fontName="Helvetica",
                          alignment=TA_CENTER, spaceAfter=10)
    s_verdict = S("vrd",  fontSize=18, textColor=verdict_color, fontName="Helvetica-Bold",
                          alignment=TA_CENTER, spaceAfter=0)
    s_prob    = S("prb",  fontSize=12, textColor=C_HDR_BG, fontName="Helvetica-Bold",
                          alignment=TA_CENTER, spaceAfter=2)
    s_h1      = S("h1",  fontSize=12, textColor=C_VIOLET,  fontName="Helvetica-Bold",
                          spaceBefore=10, spaceAfter=5)
    s_h2      = S("h2",  fontSize=10, textColor=C_TEAL,    fontName="Helvetica-Bold",
                          spaceBefore=6,  spaceAfter=3)
    s_body    = S("bd",  fontSize=9,  textColor=C_BLACK,   fontName="Helvetica",
                          leading=14, spaceAfter=4)
    s_rec_h   = S("rch", fontSize=9,  textColor=C_AMBER,   fontName="Helvetica-Bold",
                          spaceAfter=2)
    s_footer  = S("ft",  fontSize=7,  textColor=C_BLACK,   fontName="Helvetica",
                          alignment=TA_CENTER)
    s_th      = S("th",  fontSize=9,  textColor=C_WHITE,   fontName="Helvetica-Bold",
                          alignment=TA_LEFT)
    s_td_key  = S("tdk", fontSize=8,  textColor=C_VIOLET,  fontName="Helvetica-Bold",
                          alignment=TA_LEFT)
    s_td_body = S("tdb", fontSize=8,  textColor=C_BLACK,   fontName="Helvetica",
                          alignment=TA_LEFT)
    s_dis     = S("dis", fontSize=7.5, textColor=C_BLACK,  fontName="Helvetica",
                          leading=11, borderColor=C_BORDER, borderWidth=0.5,
                          borderPadding=5, backColor=C_DIS_BG)

    # Glossary cell styles (defined once, reused per row — no inline ParagraphStyle)
    s_glos_key  = S("glos_key",  fontSize=8, textColor=C_VIOLET, fontName="Helvetica-Bold",
                                 alignment=TA_LEFT)
    s_glos_body = S("glos_body", fontSize=8, textColor=C_BLACK,  fontName="Helvetica",
                                 alignment=TA_LEFT)
    s_glos_sm   = S("glos_sm",   fontSize=7.5, textColor=C_BLACK, fontName="Helvetica",
                                 alignment=TA_LEFT)

    def rule(thickness=0.6, color=C_BORDER):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceAfter=5, spaceBefore=5)

    story = []

    # ── Page header bar ──
    hdr_data = [[
        Paragraph("<b>ThermoCardial AI</b>",
                  S("hL", fontSize=14, textColor=C_WHITE, fontName="Helvetica-Bold",
                    alignment=TA_LEFT)),
        Paragraph("CARDIAC RISK DIAGNOSTIC REPORT",
                  S("hR", fontSize=8, textColor=C_BORDER, fontName="Helvetica",
                    alignment=TA_LEFT)),
    ]]
    hdr_tbl = Table(hdr_data, colWidths=[(W-44*mm)*0.45, (W-44*mm)*0.55])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_HDR_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("THERMODYNAMIC ATTENTION REGRESSION SYSTEM  |  VIT Bhopal", s_tag))
    story.append(Paragraph(f"Generated: {stamp}", s_stamp))
    story.append(rule(1.2, C_VIOLET))

    # ── Verdict banner ──
    v_data = [[Paragraph(f"<b>{'[!]' if pred==1 else '[OK]'}  {verdict_label}</b>", s_verdict)]]
    v_tbl = Table(v_data, colWidths=[W - 44*mm])
    v_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), banner_bg),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("BOX",           (0,0),(-1,-1), 1, verdict_color),
    ]))
    story.append(v_tbl)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"Risk Probability: <b>{prob:.1%}</b>   |   Severity Score: <b>{sev:.2f} / 10</b>",
        s_prob))
    story.append(rule())

    # ═══════ 1. PATIENT PARAMETERS ═══════
    story.append(Paragraph("1.  Patient Clinical Parameters", s_h1))

    rev_sex  = {v:k for k,v in SEX_MAP.items()}
    rev_cp   = {v:k for k,v in CP_MAP.items()}
    rev_fbs  = {v:k for k,v in FBS_MAP.items()}
    rev_ecg  = {v:k for k,v in RESTECG_MAP.items()}
    rev_exang= {v:k for k,v in EXANG_MAP.items()}
    rev_slope= {v:k for k,v in SLOPE_MAP.items()}
    rev_ca   = {v:k for k,v in CA_MAP.items()}
    rev_thal = {v:k for k,v in THAL_MAP.items()}

    inp_rows = [
        [Paragraph("<b>Field</b>",        s_th),
         Paragraph("<b>Full Name</b>",    s_th),
         Paragraph("<b>Value</b>",        s_th),
         Paragraph("<b>Unit / Code</b>",  s_th)],
        [Paragraph("<b>age</b>",     s_td_key), Paragraph("Age",                              s_td_body), Paragraph(str(int(inputs["age"])),                                    s_td_body), Paragraph("years",   s_td_body)],
        [Paragraph("<b>sex</b>",     s_td_key), Paragraph("Biological Sex",                   s_td_body), Paragraph(rev_sex.get(int(inputs["sex"]),"—"),                        s_td_body), Paragraph("—",       s_td_body)],
        [Paragraph("<b>cp</b>",      s_td_key), Paragraph("Chest Pain Type",                  s_td_body), Paragraph(rev_cp.get(int(inputs["cp"]),"—"),                          s_td_body), Paragraph("0-3",     s_td_body)],
        [Paragraph("<b>trestbps</b>",s_td_key), Paragraph("Resting Blood Pressure",           s_td_body), Paragraph(str(int(inputs["trestbps"])),                               s_td_body), Paragraph("mm Hg",   s_td_body)],
        [Paragraph("<b>chol</b>",    s_td_key), Paragraph("Serum Cholesterol",                s_td_body), Paragraph(str(int(inputs["chol"])),                                   s_td_body), Paragraph("mg/dl",   s_td_body)],
        [Paragraph("<b>fbs</b>",     s_td_key), Paragraph("Fasting Blood Sugar",              s_td_body), Paragraph(rev_fbs.get(int(inputs["fbs"]),"—"),                        s_td_body), Paragraph("—",       s_td_body)],
        [Paragraph("<b>restecg</b>", s_td_key), Paragraph("Resting ECG Results",              s_td_body), Paragraph(rev_ecg.get(int(inputs["restecg"]),"—"),                    s_td_body), Paragraph("0-2",     s_td_body)],
        [Paragraph("<b>thalach</b>", s_td_key), Paragraph("Max Heart Rate Achieved",          s_td_body), Paragraph(str(int(inputs["thalach"])),                                s_td_body), Paragraph("bpm",     s_td_body)],
        [Paragraph("<b>exang</b>",   s_td_key), Paragraph("Exercise Induced Angina",          s_td_body), Paragraph(rev_exang.get(int(inputs["exang"]),"—"),                    s_td_body), Paragraph("—",       s_td_body)],
        [Paragraph("<b>oldpeak</b>", s_td_key), Paragraph("ST Depression (exercise vs rest)", s_td_body), Paragraph(f"{inputs['oldpeak']:.1f}",                                 s_td_body), Paragraph("mm",      s_td_body)],
        [Paragraph("<b>slope</b>",   s_td_key), Paragraph("Peak Exercise ST Segment Slope",   s_td_body), Paragraph(rev_slope.get(int(inputs["slope"]),"—"),                    s_td_body), Paragraph("0-2",     s_td_body)],
        [Paragraph("<b>ca</b>",      s_td_key), Paragraph("Major Vessels (fluoroscopy)",      s_td_body), Paragraph(rev_ca.get(int(inputs["ca"]),"—"),                          s_td_body), Paragraph("0-4",     s_td_body)],
        [Paragraph("<b>thal</b>",    s_td_key), Paragraph("Thalassemia / Stress Test Result", s_td_body), Paragraph(rev_thal.get(int(inputs["thal"]),"—"),                      s_td_body), Paragraph("0-3",     s_td_body)],
    ]
    col_w = [(W-44*mm)*x for x in [0.13, 0.35, 0.32, 0.20]]
    inp_tbl = Table(inp_rows, colWidths=col_w)
    inp_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_HDR_BG),
        ("BACKGROUND",    (0,1),(-1,-1), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_ROW_ALT]),
        ("GRID",          (0,0),(-1,-1), 0.4, C_BORDER),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story.append(inp_tbl)
    story.append(rule())

    # ═══════ 2. CLINICAL ANALYSIS ═══════
    story.append(Paragraph("2.  Clinical Analysis", s_h1))
    age = inputs["age"]
    expected_max = int(220 - age)
    hrr    = inputs["thalach"] - expected_max
    strain = (inputs["oldpeak"] * inputs["trestbps"]) / (inputs["thalach"] + 1e-6)
    rpp    = (inputs["trestbps"] * inputs["thalach"]) / 1000
    hr_note  = "within target" if inputs["thalach"] >= 0.85*expected_max else "below 85% of target - possible chronotropic incompetence"
    hrr_note = "adequate cardiac reserve" if hrr >= 0 else "negative - suggests reduced cardiac reserve"
    analysis_lines = [
        f"<b>Age-adjusted Max HR Target:</b> {expected_max} bpm  |  Achieved: {int(inputs['thalach'])} bpm  ({hr_note})",
        f"<b>Heart Rate Reserve (HRR):</b> {hrr:+.0f} bpm  ({hrr_note})",
        f"<b>Thermodynamic Strain Index:</b> {strain:.3f}  (oldpeak x trestbps / thalach - higher = greater ischaemic burden)",
        f"<b>Rate-Pressure Product (RPP):</b> {rpp:.1f} x 10^3  ({'elevated demand - RPP >20 is concerning' if rpp > 20 else 'within acceptable myocardial oxygen demand range'})",
        f"<b>Overall Prediction:</b> {verdict_label}  |  Probability: {prob:.1%}  |  Severity: {sev:.2f}/10",
    ]
    for line in analysis_lines:
        story.append(Paragraph("  " + line, s_body))
    story.append(rule())

    # ═══════ 3. RECOMMENDATIONS ═══════
    story.append(Paragraph("3.  Field-Level Health Recommendations", s_h1))
    recs = build_health_recommendations(inputs, pred, prob)
    for field_name, tip in recs:
        story.append(Paragraph(f">> {field_name}", s_rec_h))
        story.append(Paragraph(f"   {tip}", s_body))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("General Heart Health Guidelines", s_h2))
    general = [
        "Exercise: Aim for 150+ min/week of moderate aerobic activity (walking, cycling, swimming).",
        "Diet: Follow a Mediterranean or DASH diet - vegetables, legumes, whole grains, fish, olive oil.",
        "Sleep: Maintain 7-9 hours/night. Sleep apnea significantly elevates cardiac risk.",
        "Stress: Practice mindfulness or breathing exercises. Chronic stress elevates cortisol and BP.",
        "Smoking: Quitting smoking halves cardiac risk within one year. Seek cessation support.",
        "Monitoring: Check BP and cholesterol annually. Diabetic patients test HbA1c every 3 months.",
        "Medications: Never stop prescribed cardiac medications without consulting your physician.",
    ]
    for tip in general:
        story.append(Paragraph(f"   - {tip}", s_body))
    story.append(rule())

    # ═══════ 4. GLOSSARY (page 2) ═══════
    story.append(PageBreak())
    hdr2 = Table([[Paragraph("<b>ThermoCardial AI</b>  -  Field Glossary &amp; Clinical Reference",
                             S("hdr2", fontSize=12, textColor=C_WHITE, fontName="Helvetica-Bold",
                               alignment=TA_LEFT))]], colWidths=[W - 44*mm])
    hdr2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_HDR_BG),
                               ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
                               ("LEFTPADDING",(0,0),(-1,-1),10)]))
    story.append(hdr2)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("4.  Field Glossary - Full Form &amp; Clinical Reference Ranges", s_h1))

    # ── FIX: use pre-defined styles instead of inline ParagraphStyle with parent= ──
    glos_rows = [[
        Paragraph("<b>Field</b>",         s_th),
        Paragraph("<b>Full Name</b>",     s_th),
        Paragraph("<b>Unit</b>",          s_th),
        Paragraph("<b>Dataset Range</b>", s_th),
    ]]
    for key, meta in FIELD_META.items():
        glos_rows.append([
            Paragraph(f"<b>{key}</b>",   s_glos_key),
            Paragraph(meta["full"],      s_glos_body),
            Paragraph(meta["unit"],      s_glos_sm),
            Paragraph(meta["range"],     s_glos_sm),
        ])

    g_col_w = [(W-44*mm)*x for x in [0.11, 0.33, 0.20, 0.36]]
    glos_tbl = Table(glos_rows, colWidths=g_col_w)
    glos_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_HDR_BG),
        ("BACKGROUND",    (0,1),(-1,-1), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_ROW_ALT]),
        ("GRID",          (0,0),(-1,-1), 0.4, C_BORDER),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story.append(glos_tbl)
    story.append(rule())
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system and is intended for educational "
        "and screening purposes only. It does NOT constitute medical advice, diagnosis, or treatment. "
        "Always consult a qualified cardiologist or physician for clinical decisions.",
        s_dis))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"ThermoCardial AI  |  Thermodynamic Attention Regression System  |  VIT Bhopal  |  "
        f"Report ID: TC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        s_footer))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
CHART_BG = "#0a0514"; PAPER_BG = "#130d24"
FONT_COLOR = "#c4b5fd"; GRID_COLOR = "#2e1f4a"

def plotly_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=13,color=FONT_COLOR),x=0.02),
        paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Space Grotesk",color=FONT_COLOR,size=11),
        margin=dict(l=40,r=20,t=40,b=40),
        xaxis=dict(gridcolor=GRID_COLOR,linecolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR,linecolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
    )

def patient_radar(inputs):
    cats = ["Age","Resting BP","Cholesterol","Max HR","ST Depress.","Vessels","Thalassemia","Chest Pain"]
    maxv = [80, 200, 564, 210, 6.5, 4, 3, 3]
    vals = [inputs["age"],inputs["trestbps"],inputs["chol"],inputs["thalach"],
            inputs["oldpeak"],inputs["ca"],inputs["thal"],inputs["cp"]]
    norm = [v/m for v,m in zip(vals,maxv)]
    norm_c = norm + [norm[0]]; cats_c = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm_c, theta=cats_c, fill='toself',
        fillcolor='rgba(124,58,237,0.22)',
        line=dict(color='#a78bfa',width=2), name="Patient"))
    fig.update_layout(
        polar=dict(bgcolor=CHART_BG,
            radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID_COLOR,
                            linecolor=GRID_COLOR,tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID_COLOR,linecolor=GRID_COLOR)),
        paper_bgcolor=PAPER_BG,
        font=dict(family="Space Grotesk",color=FONT_COLOR,size=11),
        margin=dict(l=50,r=50,t=30,b=30), showlegend=False, height=320)
    return fig

def patient_bar(inputs):
    fields = {"Age":inputs["age"],"Rest BP":inputs["trestbps"],"Cholesterol":inputs["chol"],
              "Max HR":inputs["thalach"],"ST Depress×30":inputs["oldpeak"]*30,
              "Vessels×50":inputs["ca"]*50,"Thal×50":inputs["thal"]*50,"Chest Pain×30":inputs["cp"]*30}
    fig = go.Figure(go.Bar(x=list(fields.keys()), y=list(fields.values()),
        marker=dict(color=list(fields.values()),
                    colorscale=[[0,"#4f46e5"],[0.5,"#7c3aed"],[1,"#ec4899"]],
                    line=dict(width=0))))
    fig.update_layout(**plotly_layout("Clinical Parameters (raw / scaled for display)"), height=280)
    return fig

def gauge_chart(prob):
    color = "#ef4444" if prob >= 0.5 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number=dict(suffix="%",font=dict(size=28,color=color,family="JetBrains Mono")),
        gauge=dict(axis=dict(range=[0,100],tickfont=dict(color=FONT_COLOR,size=9)),
                   bar=dict(color=color,thickness=0.25), bgcolor=CHART_BG,
                   bordercolor=GRID_COLOR,
                   steps=[dict(range=[0,40],color="rgba(16,185,129,.12)"),
                          dict(range=[40,60],color="rgba(251,191,36,.10)"),
                          dict(range=[60,100],color="rgba(239,68,68,.12)")],
                   threshold=dict(line=dict(color=color,width=3),thickness=0.75,value=prob*100))))
    fig.update_layout(paper_bgcolor=PAPER_BG,
        font=dict(family="Space Grotesk",color=FONT_COLOR),
        margin=dict(l=20,r=20,t=20,b=10), height=210)
    return fig

def community_donut(pos, neg):
    fig = go.Figure(go.Pie(labels=["High Risk","Low Risk"], values=[pos,neg], hole=0.62,
        marker=dict(colors=["#ef4444","#10b981"],line=dict(color=PAPER_BG,width=2)),
        textfont=dict(color="#fff",size=12)))
    fig.update_layout(paper_bgcolor=PAPER_BG,
        font=dict(family="Space Grotesk",color=FONT_COLOR),
        margin=dict(l=10,r=10,t=20,b=20), showlegend=True,
        legend=dict(font=dict(color=FONT_COLOR,size=11),orientation="h",x=0.1,y=-0.05),
        height=260)
    return fig

def trend_line(df_hist):
    if df_hist.empty: return None
    df_hist = df_hist.sort_values("id").copy()
    df_hist["cumulative_positive"] = (df_hist["prediction"]==1).cumsum()
    df_hist["cumulative_negative"] = (df_hist["prediction"]==0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist["id"],y=df_hist["cumulative_positive"],
        name="High Risk",line=dict(color="#ef4444",width=2),
        fill='tozeroy',fillcolor='rgba(239,68,68,0.08)'))
    fig.add_trace(go.Scatter(x=df_hist["id"],y=df_hist["cumulative_negative"],
        name="Low Risk",line=dict(color="#10b981",width=2),
        fill='tozeroy',fillcolor='rgba(16,185,129,0.08)'))
    fig.update_layout(**plotly_layout("Cumulative Predictions Over Time"),height=220,
        legend=dict(orientation="h",x=0.02,y=.97,font=dict(color=FONT_COLOR,size=11)))
    return fig

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
init_db()
artifacts, missing = load_predictor()

# ── Hero ──
st.markdown("""
<div class="hero">
  <p class="hero-title" style="font-size: 3rem; font-weight: 900;">🫀 <span style="color:white;">ThermoCardial AI</span></p>
  <p class="hero-sub">Thermodynamic Attention Regression · Cardiac Risk Intelligence</p>
</div>
""", unsafe_allow_html=True)

if missing:
    st.error(f"⚠️ Missing model files: `{', '.join(missing)}`\n\n"
             "Place your `.pkl` files in the same directory as `app.py` and restart.")
    st.stop()

# ════════════════════════════════════════════
# SECTION 1 — INPUT FORM
# ════════════════════════════════════════════
st.markdown('<p class="section-label">Diagnostic Input</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Patient Clinical Parameters</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="col-heading">Demographics &amp; Vitals</p>', unsafe_allow_html=True)

        age = st.number_input("Age (years)",
            min_value=1, max_value=120, value=52,
            help=FIELD_META["age"]["tip"])

        sex_label = st.selectbox("Sex",
            options=list(SEX_MAP.keys()),
            help=FIELD_META["sex"]["tip"])
        sex = SEX_MAP[sex_label]

        trestbps = st.number_input("Trestbps — Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=250, value=130,
            help=FIELD_META["trestbps"]["tip"])

        chol = st.number_input("Chol — Serum Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=220,
            help=FIELD_META["chol"]["tip"])

        fbs_label = st.selectbox("FBS — Fasting Blood Sugar",
            options=list(FBS_MAP.keys()),
            help=FIELD_META["fbs"]["tip"])
        fbs = FBS_MAP[fbs_label]

    with col2:
        st.markdown('<p class="col-heading">Cardiac Metrics</p>', unsafe_allow_html=True)

        cp_label = st.selectbox("CP — Chest Pain Type",
            options=list(CP_MAP.keys()),
            help=FIELD_META["cp"]["tip"])
        cp = CP_MAP[cp_label]

        restecg_label = st.selectbox("Restecg — Resting ECG Results",
            options=list(RESTECG_MAP.keys()),
            help=FIELD_META["restecg"]["tip"])
        restecg = RESTECG_MAP[restecg_label]

        thalach = st.number_input("Thalach — Max Heart Rate Achieved (bpm)",
            min_value=60, max_value=250, value=150,
            help=FIELD_META["thalach"]["tip"])

        exang_label = st.selectbox("Exang — Exercise Induced Angina",
            options=list(EXANG_MAP.keys()),
            help=FIELD_META["exang"]["tip"])
        exang = EXANG_MAP[exang_label]

    with col3:
        st.markdown('<p class="col-heading">ST Segment &amp; Vessels</p>', unsafe_allow_html=True)

        oldpeak = st.number_input("Oldpeak — ST Depression (mm)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f",
            help=FIELD_META["oldpeak"]["tip"])

        slope_label = st.selectbox("Slope — Peak ST Segment Slope",
            options=list(SLOPE_MAP.keys()),
            help=FIELD_META["slope"]["tip"])
        slope = SLOPE_MAP[slope_label]

        ca_label = st.selectbox("CA — Major Vessels (fluoroscopy)",
            options=list(CA_MAP.keys()),
            help=FIELD_META["ca"]["tip"])
        ca = CA_MAP[ca_label]

        thal_label = st.selectbox("Thal — Thalassemia / Stress Test",
            options=list(THAL_MAP.keys()),
            help=FIELD_META["thal"]["tip"])
        thal = THAL_MAP[thal_label]

    st.markdown("")
    submitted = st.form_submit_button("🔍  Analyse Cardiac Risk")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "result" not in st.session_state: st.session_state.result = None
if "inputs" not in st.session_state: st.session_state.inputs = None

if submitted:
    inputs_dict = dict(age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,
                       fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,
                       oldpeak=oldpeak,slope=slope,ca=ca,thal=thal)
    patient_vals = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

    with st.spinner("Running Thermodynamic Attention Regression…"):
        result = run_prediction(artifacts, patient_vals)

    save_prediction(inputs_dict, result["prediction"], result["probability"], result["severity"])
    st.session_state.result = result
    st.session_state.inputs = inputs_dict

# ════════════════════════════════════════════
# SECTION 2 — RESULT + CHARTS
# ════════════════════════════════════════════
if st.session_state.result:
    result = st.session_state.result
    inputs = st.session_state.inputs
    prob = result["probability"]; pred = result["prediction"]; sev = result["severity"]

    st.markdown("---")
    st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Diagnostic Report</p>', unsafe_allow_html=True)

    r_col, g_col = st.columns([1, 1.6])

    with r_col:
        if pred == 1:
            st.markdown(f"""
            <div class="result-positive">
              <div class="result-icon">⚠️</div>
              <div class="result-label" style="color:#f87171;">High Cardiac Risk</div>
              <div class="result-prob">Probability · {prob:.4f} &nbsp;|&nbsp; Severity · {sev:.2f}/10</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
              <div class="result-icon">✅</div>
              <div class="result-label" style="color:#34d399;">Low Cardiac Risk</div>
              <div class="result-prob">Probability · {prob:.4f} &nbsp;|&nbsp; Severity · {sev:.2f}/10</div>
            </div>""", unsafe_allow_html=True)

        st.plotly_chart(gauge_chart(prob), width="stretch", config={"displayModeBar":False})

        # ── PDF Download ──
        st.markdown("---")
        st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(inputs, result)
        fname = f"ThermoCardial_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="📄  Download Full PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )

    with g_col:
        st.plotly_chart(patient_radar(inputs), width="stretch", config={"displayModeBar":False})

    st.plotly_chart(patient_bar(inputs), width="stretch", config={"displayModeBar":False})

    with st.expander("📋 Input Parameter Summary"):
        labels = {"age":"Age","sex":"Sex","cp":"Chest Pain Type","trestbps":"Resting BP (mm Hg)",
                  "chol":"Cholesterol (mg/dl)","fbs":"Fasting BS > 120","restecg":"Resting ECG",
                  "thalach":"Max Heart Rate","exang":"Exercise Angina","oldpeak":"ST Depression",
                  "slope":"ST Slope","ca":"Major Vessels","thal":"Thalassemia"}
        tbl = pd.DataFrame({"Parameter":list(labels.values()),
                            "Value":[inputs[k] for k in labels]})
        st.dataframe(tbl, width="stretch", hide_index=True)

# ════════════════════════════════════════════
# SECTION 3 — COMMUNITY STATISTICS
# ════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-label">Population Insights</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Community Prediction Statistics</p>', unsafe_allow_html=True)

hist_df = load_all_predictions()
total = len(hist_df)

if total == 0:
    st.info("No predictions recorded yet. Submit a patient form above to populate the dashboard.")
else:
    n_pos = int((hist_df["prediction"]==1).sum())
    n_neg = int((hist_df["prediction"]==0).sum())
    avg_prob = hist_df["probability"].mean()

    t1,t2,t3,t4 = st.columns(4)
    with t1:
        st.markdown(f"""<div class="stat-tile stat-total">
            <div class="stat-num" style="color:#818cf8;">{total}</div>
            <div class="stat-desc">Total Predictions</div></div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""<div class="stat-tile stat-positive">
            <div class="stat-num" style="color:#f87171;">{n_pos}</div>
            <div class="stat-desc">High Risk Cases</div></div>""", unsafe_allow_html=True)
    with t3:
        st.markdown(f"""<div class="stat-tile stat-negative">
            <div class="stat-num" style="color:#34d399;">{n_neg}</div>
            <div class="stat-desc">Low Risk Cases</div></div>""", unsafe_allow_html=True)
    with t4:
        st.markdown(f"""<div class="stat-tile stat-total">
            <div class="stat-num" style="color:#67e8f9;">{avg_prob:.2%}</div>
            <div class="stat-desc">Avg Risk Probability</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    d_col, tr_col = st.columns([1, 1.8])
    with d_col:
        st.markdown("**Risk Distribution**")
        st.plotly_chart(community_donut(n_pos,n_neg), width="stretch", config={"displayModeBar":False})
    with tr_col:
        st.markdown("**Cumulative Trend**")
        trend = trend_line(hist_df)
        if trend: st.plotly_chart(trend, width="stretch", config={"displayModeBar":False})

    fig_hist = px.histogram(hist_df, x="probability", nbins=20,
        color_discrete_sequence=["#7c3aed"],
        labels={"probability":"Predicted Risk Probability"})
    fig_hist.update_layout(**plotly_layout("Distribution of Risk Probabilities"), height=220)
    fig_hist.update_traces(marker_line_width=0)
    st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar":False})

    with st.expander(f"🗂️  Recent Prediction Records (last 10 of {total})"):
        display_cols = ["id","timestamp","age","sex","cp","trestbps","chol",
                        "thalach","exang","oldpeak","prediction","probability","severity"]
        st.dataframe(hist_df[display_cols].head(10).rename(columns={"id":"#","timestamp":"Time"}),
                     width="stretch", hide_index=True)

st.markdown("""<br><hr>
<p style='text-align:center;color:#4a3d6b;font-size:.75rem;letter-spacing:1px;'>
ThermoCardial AI · Thermodynamic Attention Regression System · VIT Bhopal
</p>""", unsafe_allow_html=True)
"""
=============================================================================
AGENT 4: STREAMLIT DEPLOYMENT APP
=============================================================================
Breast Cancer Detection - Interactive Prediction System
Premium dark-themed UI with glassmorphism effects.

Run with:  streamlit run app.py
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detector | GENOMEX",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model & scaler
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model files not found! Run `python main.py` first to train and save the model.")
    st.stop()

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature metadata
data = load_breast_cancer()
feature_names = data.feature_names.tolist()

# Pre-compute dataset statistics for slider ranges
df_stats = {
    name: {"min": round(float(data.data[:, i].min()), 4),
           "max": round(float(data.data[:, i].max()), 4),
           "mean": round(float(data.data[:, i].mean()), 4)}
    for i, name in enumerate(feature_names)
}

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero .subtitle {
        font-size: 1.1rem;
        color: #999;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(26, 29, 35, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
    }

    /* Result cards */
    .result-benign {
        background: linear-gradient(135deg, rgba(0,201,167,0.15), rgba(0,201,167,0.05));
        border: 1px solid rgba(0,201,167,0.4);
        border-left: 4px solid #00C9A7;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-malignant {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,107,107,0.05));
        border: 1px solid rgba(255,107,107,0.4);
        border-left: 4px solid #FF6B6B;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Sample data button styling */
    .sample-btn {
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #555;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>BREAST CANCER DETECTOR</h1>
    <p class="subtitle">
        AI-Powered Diagnostic Assistant &nbsp;|&nbsp; Wisconsin Breast Cancer Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        "This ML-powered tool analyses **30 cell nucleus measurements** "
        "from fine-needle aspirate (FNA) images to predict whether a "
        "breast tumour is **Benign** or **Malignant**."
    )

    st.markdown("---")

    st.markdown("### Model Details")
    col1, col2 = st.columns(2)
    col1.metric("Algorithm", type(model).__name__)
    col2.metric("Features", "30")

    st.markdown("")
    st.markdown("**Training Data**: 569 samples")
    st.markdown("**Priority**: High Recall")
    st.markdown("**Preprocessing**: StandardScaler")

    st.markdown("---")

    st.markdown("### Quick Actions")
    if st.button("Load Benign Sample", use_container_width=True, key="btn_benign"):
        st.session_state["sample_type"] = "benign"
    if st.button("Load Malignant Sample", use_container_width=True, key="btn_malignant"):
        st.session_state["sample_type"] = "malignant"
    if st.button("Reset to Defaults", use_container_width=True, key="btn_reset"):
        st.session_state["sample_type"] = "default"

    st.markdown("---")
    st.caption("For educational purposes only. Not a medical device.")

# ---------------------------------------------------------------------------
# Determine sample values
# ---------------------------------------------------------------------------
sample_type = st.session_state.get("sample_type", "default")

# Sample indices from the dataset
benign_sample = data.data[data.target == 1][0]      # first benign sample
malignant_sample = data.data[data.target == 0][0]    # first malignant sample

def get_default_value(i):
    if sample_type == "benign":
        return float(benign_sample[i])
    elif sample_type == "malignant":
        return float(malignant_sample[i])
    else:
        return df_stats[feature_names[i]]["mean"]

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown("### Enter Tumour Measurements")

groups = [
    ("Mean Values", "Mean of cell nucleus measurements", feature_names[0:10], "0"),
    ("Standard Error", "Variation in cell measurements", feature_names[10:20], "1"),
    ("Worst Values", "Largest / most extreme measurements", feature_names[20:30], "2"),
]

input_values = {}
tabs = st.tabs([f"  {g[0]}  " for g in groups])

for tab, (group_name, group_desc, features, _) in zip(tabs, groups):
    with tab:
        st.caption(group_desc)
        cols = st.columns(2)
        for j, feat in enumerate(features):
            idx = feature_names.index(feat)
            stats = df_stats[feat]
            with cols[j % 2]:
                input_values[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=stats["max"] * 2.0,
                    value=round(get_default_value(idx), 4),
                    format="%.4f",
                    key=f"input_{feat}",
                )

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    predict_clicked = st.button(
        "Analyse & Predict",
        use_container_width=True,
        type="primary",
    )

if predict_clicked:
    # Build feature vector
    feature_vector = np.array([[input_values[f] for f in feature_names]])
    scaled_vector = scaler.transform(feature_vector)
    prediction = model.predict(scaled_vector)[0]

    st.markdown("")

    if prediction == "Benign":
        st.markdown(
            '<div class="result-benign">'
            '<h2 style="color:#00C9A7; margin:0;">BENIGN</h2>'
            '<p style="color:#aaa; margin:0.5rem 0 0;">The tumour is predicted to be <b>non-cancerous</b>.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.balloons()
    else:
        st.markdown(
            '<div class="result-malignant">'
            '<h2 style="color:#FF6B6B; margin:0;">MALIGNANT</h2>'
            '<p style="color:#aaa; margin:0.5rem 0 0;">The tumour is predicted to be <b>cancerous</b>. '
            'Please consult a medical professional.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.snow()

    # --- Feature summary ---
    st.markdown("")
    with st.expander("View input feature values", expanded=False):
        display_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": [input_values[f] for f in feature_names],
            "Scaled": scaled_vector[0],
            "Dataset Mean": [df_stats[f]["mean"] for f in feature_names],
        })
        display_df["vs Mean"] = display_df.apply(
            lambda r: "ABOVE" if r["Value"] > r["Dataset Mean"] else "below", axis=1
        )
        st.dataframe(display_df.set_index("Feature"), use_container_width=True)

    st.caption("This is an educational ML demo. Not a substitute for professional medical diagnosis.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">'
    'Built with Streamlit &nbsp;|&nbsp; scikit-learn &nbsp;|&nbsp; '
    'Wisconsin Diagnostic Breast Cancer Dataset'
    '</div>',
    unsafe_allow_html=True,
)

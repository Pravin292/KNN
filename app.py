import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="KNN Elite | Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (GLASSMORPHISM) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
        color: #e0e0e0;
    }

    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Prediction Result Card */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .result-malignant {
        background: rgba(255, 75, 75, 0.2);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    .result-benign {
        background: rgba(46, 204, 113, 0.2);
        border: 1px solid #2ecc71;
        color: #2ecc71;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 52, 96, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_assets():
    model_path = "knn_model.pkl"
    scaler_path = "scaler (2).pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("🚨 Critical Error: Model or Scaler files not found!")
        return None, None
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"🚨 Failed to load assets: {e}")
        return None, None

model, scaler = load_model_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d2ff;'>💎 Elite Edition</h2>", unsafe_allow_html=True)
    st.info("K-Nearest Neighbors (KNN) Diagnostic Engine for Breast Cancer Classification.")
    st.markdown("---")
    st.write("**Model Version:** 2.1.0")
    st.write("**Engine:** Scikit-Learn")
    st.write("**Architecture:** Elite Glassmorphism")

# --- MAIN UI ---
st.markdown("<h1 class='main-title'>Clinical KNN Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892b0;'>Precision AI-driven diagnostic assistant for oncological assessment.</p>", unsafe_allow_html=True)

if model and scaler:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧬 Patient Diagnostic Parameters")
    
    cols = st.columns(3)
    
    # 30 Features of Breast Cancer Wisconsin Dataset
    feature_names = [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
        "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
        "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
        "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
        "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
        "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
    ]
    
    inputs = []
    
    # Organize into 3 columns (10 rows each)
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            # Use number_input for precision, default to some reasonable middle values or 0
            val = st.number_input(name, value=0.0, format="%.4f")
            inputs.append(val)
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- PREDICTION LOGIC ---
    if st.button("🚀 Execute Diagnostic Analysis", use_container_width=True):
        input_array = np.array([inputs])
        
        # Scaling
        try:
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            
            st.markdown("---")
            st.subheader("⚡ Diagnostic Results")
            
            if prediction[0] == 0:
                st.markdown("""
                <div class='result-card result-malignant'>
                    ⚠️ DIAGNOSIS: MALIGNANT<br>
                    High-risk indicators detected. Urgent clinical review recommended.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-card result-benign'>
                    ✅ DIAGNOSIS: BENIGN<br>
                    Low-risk indicators. Non-cancerous patterns detected.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Inference Error: {e}")

else:
    st.warning("Please ensure 'knn_model.pkl' and 'scaler (2).pkl' are in the project root.")

# --- FOOTER ---
st.markdown("<br><hr><center><p style='color:#4b5563; font-size:0.8rem;'>Pravin292 Intelligence Systems © 2026</p></center>", unsafe_allow_html=True)
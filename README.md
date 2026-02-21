# 🧬 KNN Elite: Clinical Diagnostic Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced oncological diagnostic assistant powered by the **K-Nearest Neighbors (KNN)** algorithm. This "Elite Edition" features a high-fidelity **Cyber-Glassmorphism** interface designed for precision medical analytics.

## ✨ Elite Features

- **💎 Midnight Glass UI**: A premium, dark-mode aesthetic with glassmorphic components and fluid animations.
- **🧬 30-Point Diagnostic Array**: Full integration of the Wisconsin Breast Cancer dataset features for comprehensive analysis.
- **⚡ Real-time Inference**: Instance-based classification between Malignant (High-Risk) and Benign (Low-Risk) cases.
- **🎯 Precision Inputs**: Replaced imprecise sliders with direct numeric entry for clinical accuracy.
- **📊 Modular Architecture**: Clean separation of UI logic, model serialization, and data preprocessing.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Pravin292/KNN.git
cd KNN
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 🧠 Technical Overview

### The Algorithm: K-Nearest Neighbors (KNN)
The engine utilizes a supervised learning approach that classifies new data points based on their similarity to existing labeled cases in the feature space.

### Dataset
Built upon the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
- **Samples**: 569
- **Features**: 30 numeric, predictive attributes
- **Classes**: Malignant (0), Benign (1)

### Accuracy & Scaling
- **Standardization**: Features are scaled using `StandardScaler` to ensure distance-based metrics are not biased by magnitude.
- **Validation**: Optimized for high recall to minimize false negatives in clinical scenarios.

## 🛠 Project Structure

- `app.py`: The main Streamlit dashboard and UI logic.
- `knn_model.pkl`: Serialized KNN classifier.
- `scaler (2).pkl`: Pre-fitted StandardScaler for input normalization.
- `requirements.txt`: Environment dependencies.

---

**Disclaimer**: This tool is for educational and research purposes only and should not be used as a primary diagnostic tool without clinical validation.

Developed with ❤️ by [Pravin292](https://github.com/Pravin292)

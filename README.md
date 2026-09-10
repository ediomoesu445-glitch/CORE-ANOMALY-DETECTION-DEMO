# CORE — Anomaly Detection System

**Machine Learning-powered fault detection for petroleum process facilities**  
***CORE** — Cognitive Operations and Risk Engine for Oil and Gas Industries*

---

## Overview

This project demonstrates an AI-driven anomaly detection system built on the **Tennessee Eastman Process (TEP)** dataset — an industry-standard benchmark that simulates a real chemical/petroleum process plant with 52 process variables and 20 distinct fault types.

The system uses a **two-tier detection architecture**:

| Tier | Method | Role |
|------|--------|------|
| **1 — Screener** | PCA-MSPC (T² + Q statistics) | Detects *any* deviation from normal operation in real-time |
| **2 — Classifier** | LightGBM (Gradient Boosting) | Confirms the fault and identifies its type with 99.9% accuracy |

---

## Live Demo

The Streamlit dashboard provides:
- **Live Simulation** — replay any of 20 fault scenarios with animated sensor charts and real-time alarm indicators
- **Model Performance** — interactive comparison of 4 ML models (Logistic Regression, Random Forest, LightGBM, MLP Neural Network)
- **Per-Fault Analysis** — detection rate breakdown across all 20 fault types
- **MSPC Control Charts** — T² and Q statistics with UCL alarm lines and contribution plots
- **Methodology** — full technical walkthrough and deployment roadmap

---

## Key Results

| Metric | Value |
|--------|-------|
| Fault Detection Rate (LightGBM) | **99.9%** |
| False Alarm Rate | **0.03%** |
| ROC-AUC | **0.9997** |
| F1 Score | **0.9983** |
| Faults Detected | 20 distinct types |
| Process Variables Monitored | 52 sensors simultaneously |
| Training Time (LightGBM) | 57 seconds |

---

## Project Structure

```
├── app.py                          # Streamlit dashboard (main application)
├── save_models.py                  # Train and save LightGBM + PCA models
├── prep_demo_data.py               # Pre-split large CSVs into per-fault files
├── requirements.txt                # Python dependencies
├── Launch App.bat                  # Windows one-click launcher
│
├── scaler.pkl                      # Fitted StandardScaler
├── lgbm_model.pkl                  # Trained LightGBM classifier
├── pca_model.pkl                   # Fitted PCA + MSPC parameters
├── feature_cols.txt                # Feature column names
│
├── tep_model_summary.csv           # Overall model performance metrics
├── tep_per_fault_fdr.csv           # Per-fault detection rates (all models)
├── mspc_vs_ml_summary.csv          # MSPC vs ML comparison
├── mspc_per_fault.csv              # MSPC per-fault performance
│
├── assets/                         # Pre-generated charts and figures
│   ├── fig1_overall_metrics.png
│   ├── fig2_speed_vs_fdr.png
│   ├── ...
│   └── contribution_fault_20.png
│
├── CORE_Anomaly_Detection_Presentation.html     # Executive slide deck (HTML)
└── TEP_Fault_Detection_Report.html              # Full analysis report (HTML)
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ediomoesu445-glitch/CORE-ANOMALY-DETECTION-DEMO.git
cd CORE-ANOMALY-DETECTION-DEMO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the TEP dataset
Download the Tennessee Eastman Process dataset from:  
**Kaggle:** [Tennessee Eastman Process Simulation Dataset](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset)

Place the four CSV files in the project root:
- `TEP_FaultFree_Training.csv`
- `TEP_Faulty_Training.csv`
- `TEP_FaultFree_Testing.csv`
- `TEP_Faulty_Testing.csv`

### 4. Train the models
```bash
python save_models.py
```
This takes approximately 60 seconds and saves `scaler.pkl`, `lgbm_model.pkl`, `pca_model.pkl`, and `feature_cols.txt`.

### 5. Pre-split the data (for fast demo loading)
```bash
python prep_demo_data.py
```
This splits the 3.5 GB testing file into 20 small per-fault files in `fault_data/`, reducing load time from ~3 minutes to ~1 second.

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Or on Windows, double-click **`Launch App.bat`**.

Open your browser at **http://localhost:8501**

---

## How It Works

### Stage 1 — PCA-MSPC Screening
- StandardScaler fitted on 250,000 fault-free training samples
- PCA reduces 52 dimensions → 31 principal components (90.1% variance explained)
- **T² statistic**: detects movement within the normal operating space
- **Q statistic (SPE)**: detects movement *outside* the normal subspace
- Control limits at 99% confidence: T² UCL = 52.2, Q UCL = 11.98

### Stage 2 — LightGBM Classification
- Trained on 500,000 balanced samples (250k normal + 250k faulty)
- 372 boosting rounds, trained in 57 seconds
- Outputs fault probability score (0–1) for every sample
- Binary classification: Normal (0) vs Fault (1)

### Contribution Plots
When an alarm fires, the system automatically computes which of the 52 sensors is contributing most to the T² and Q statistics — pinpointing the root cause in seconds.

---

## Industry Context

This system was designed with Nigeria's petroleum midstream and downstream sector in mind:

- **Pipeline leak detection** — pressure/flow anomalies indicating ruptures
- **LPG plant overpressure** — cooling water and pressure vessel faults
- **Petroleum depot monitoring** — tank level and sensor drift detection
- **Gas plant compressor health** — early bearing wear and seal degradation

See `CORE_Anomaly_Detection_Presentation.html` for the full executive briefing.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | LightGBM 4.x |
| MSPC | scikit-learn PCA + SciPy |
| Dashboard | Streamlit 1.35+ |
| Data Processing | Pandas 2.x, NumPy |
| Visualisation | Matplotlib |
| Language | Python 3.10+ |

---

## Author

**Ediomo Esu**  
CORE — Cognitive Operations and Risk Engine for Oil and Gas Industries  

---

## Dataset Citation

Rieth, C.A., Amsel, B.D., Tran, R., & Cook, M.B. (2017). *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse.

---

## License

This project is developed for regulatory and research purposes.  
The Tennessee Eastman Process dataset is publicly available for research use.

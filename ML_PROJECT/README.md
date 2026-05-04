<p align="center">
  <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" alt="logo" width="80"/>
</p>

<h1 align="center">Breast Cancer Detection</h1>

<p align="center">
  <em>AI-powered diagnostic assistant using the Wisconsin Breast Cancer dataset</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-00C9A7?style=for-the-badge" alt="Status"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-97.4%25-brightgreen?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Recall-92.9%25-blue?style=flat-square" alt="Recall"/>
  <img src="https://img.shields.io/badge/Precision-100%25-blueviolet?style=flat-square" alt="Precision"/>
  <img src="https://img.shields.io/badge/F1_Score-96.3%25-orange?style=flat-square" alt="F1"/>
</p>

---

## Overview

This project predicts whether a breast tumour is **Benign** (non-cancerous) or **Malignant** (cancerous) based on 30 cell-nucleus measurements extracted from fine-needle aspirate (FNA) images.

> **Why Recall Matters**: In cancer diagnosis, a **false negative** (missing a malignant tumour) is far more dangerous than a false positive. This system prioritises **Recall** to minimise missed diagnoses.

The pipeline is built with a **multi-agent architecture** where each stage has clearly defined inputs, outputs, and responsibilities.

---

## Architecture

```
                    ┌─────────────────┐
                    │    main.py      │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │   Agent 1     │  │   Agent 2     │  │   Agent 3     │
  │ Data & Preproc│─▶│   EDA &       │  │ Model Train & │
  │               │  │   Analysis    │  │  Evaluation   │
  └───────┬───────┘  └───────────────┘  └───────┬───────┘
          │                                     │
          │              ┌──────────────┐       │
          └──────────────▶   Agent 4    ◀───────┘
                         │ Streamlit App│
                         └──────────────┘
```

---

## Project Structure

```
ML PROJECT/
│
├── data/
│   └── data_loader.py             # Agent 1 — Data loading, validation & preprocessing
│
├── notebooks/
│   ├── eda_analysis.py            # Agent 2 — Exploratory Data Analysis
│   └── plots/                     # Auto-generated visualisations (10 plots)
│       ├── 01_class_distribution.png
│       ├── 02_feature_correlations.png
│       ├── 03_correlation_heatmap.png
│       ├── 04_violin_plots_top6.png
│       ├── 05_histograms_kde_top6.png
│       ├── 06_pair_plot_top4.png
│       ├── 07_feature_importance.png
│       ├── 08_model_comparison.png
│       ├── 09_confusion_matrices.png
│       └── 10_feature_importance_lr.png
│
├── models/
│   ├── train.py                   # Agent 3 — Model training & evaluation
│   ├── best_model.pkl             # Saved best model (generated)
│   └── scaler.pkl                 # Saved StandardScaler (generated)
│
├── app.py                         # Agent 4 — Streamlit web application
├── main.py                        # Pipeline orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Dataset

| Property | Details |
|:---|:---|
| **Name** | Wisconsin Diagnostic Breast Cancer (WDBC) |
| **Source** | `sklearn.datasets.load_breast_cancer` |
| **Samples** | 569 |
| **Features** | 30 numeric (mean, SE, worst × 10 measurements) |
| **Classes** | Benign — 357 (62.7%) · Malignant — 212 (37.3%) |
| **Missing Values** | None |
| **Task** | Binary classification |

<details>
<summary><strong>Feature Groups (click to expand)</strong></summary>

Each of the 10 base measurements has three computed variants:

| # | Base Measurement | Mean | Std Error | Worst |
|---|---|---|---|---|
| 1 | Radius | mean radius | radius error | worst radius |
| 2 | Texture | mean texture | texture error | worst texture |
| 3 | Perimeter | mean perimeter | perimeter error | worst perimeter |
| 4 | Area | mean area | area error | worst area |
| 5 | Smoothness | mean smoothness | smoothness error | worst smoothness |
| 6 | Compactness | mean compactness | compactness error | worst compactness |
| 7 | Concavity | mean concavity | concavity error | worst concavity |
| 8 | Concave Points | mean concave points | concave points error | worst concave points |
| 9 | Symmetry | mean symmetry | symmetry error | worst symmetry |
| 10 | Fractal Dimension | mean fractal dimension | fractal dimension error | worst fractal dimension |

</details>

---

## Pipeline Steps

### Agent 1: Data & Preprocessing

- Load dataset from sklearn
- Build DataFrame with human-readable labels (`Benign` / `Malignant`)
- Validate: no missing values, all numeric features confirmed
- **StandardScaler** fitted on training data only (prevents data leakage)
- **80/20 stratified split** with `random_state=42` for reproducibility

### Agent 2: EDA & Feature Analysis

Generates **7 publication-quality visualisations** with a unified dark theme:

| # | Plot | Purpose |
|---|---|---|
| 1 | Donut + Bar chart | Class distribution overview |
| 2 | Horizontal bar chart | Feature correlation rankings |
| 3 | Heatmap | Inter-feature correlation structure |
| 4 | Violin + Strip plots | Distribution shape by class |
| 5 | Histogram + KDE | Density comparison by class |
| 6 | Pair plot | Multi-feature separability |
| 7 | Importance bar chart | Top predictive features |

**Key finding**: "Worst" measurements (worst concave points, worst perimeter, worst radius) are the strongest predictors.

### Agent 3: Model Training & Evaluation

Three models trained and evaluated with focus on **Recall**:

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| **SVM (RBF)** | **0.9737** | **1.0000** | **0.9286** | **0.9630** |
| KNN (k=5) | 0.9561 | 0.9744 | 0.9048 | 0.9383 |

> **Winner: SVM (RBF)** — tied on Recall, wins on F1-Score with perfect Precision.

Generates 3 additional plots: model comparison bar chart, confusion matrices, and feature importance from Logistic Regression coefficients.

### Agent 4: Deployment

- Premium **Streamlit web app** with dark-themed UI
- Interactive number inputs for all 30 features
- Quick-load buttons for sample Benign/Malignant data
- Colour-coded prediction results with feature summary expander
- Model + scaler serialised with `joblib`

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### 1. Clone & Install

```bash
cd "ML PROJECT"
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This executes Agent 1 → 2 → 3 sequentially, trains models, generates all 10 plots, and saves the best model.

### 3. Launch the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Individual Agents

```bash
python data/data_loader.py          # Agent 1 only
python notebooks/eda_analysis.py    # Agent 2 only
python models/train.py              # Agent 3 only
```

---

## Tech Stack

| Category | Technologies |
|:---|:---|
| **Language** | Python 3.9+ |
| **Data** | pandas, numpy |
| **ML** | scikit-learn (LogisticRegression, SVC, KNeighborsClassifier) |
| **Visualisation** | matplotlib, seaborn |
| **Serialisation** | joblib |
| **Web App** | Streamlit |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Disclaimer

> **This project is for educational purposes only.** It is not a certified medical diagnostic tool and should never be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

<p align="center">
  <sub>Built with Python, scikit-learn & Streamlit</sub>
</p>

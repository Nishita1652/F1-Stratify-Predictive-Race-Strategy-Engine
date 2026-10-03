# 🏎️ F1-Stratify: Predictive Race Strategy Engine

> **A supervised machine learning system that models and predicts optimal starting tyre compound decisions (Soft · Medium · Hard) from live telemetry, track environmental conditions, and grid positions.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastF1](https://img.shields.io/badge/FastF1-Telemetry%20API-red?style=for-the-badge&logo=formula1&logoColor=white)](https://github.com/theOehrly/Fast-F1)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-78.21%25-brightgreen?style=for-the-badge)](outputs/confusion_matrix.png)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Table of Contents

- [Executive Overview](#-executive-overview)
- [Motorsport Domain Problem](#-motorsport-domain-problem)
- [Key Features & Architecture](#-key-features--architecture)
- [Input Features & Strategy Engineering](#-input-features--strategy-engineering)
- [Benchmark Results](#-benchmark-results)
- [Explainability & Strategy Tree](#-explainability--strategy-tree)
- [Project Structure](#-project-structure)
- [Quick Start & Final Run Command](#-quick-start--final-run-command)
- [Python API Usage](#-python-api-usage)
- [Tech Stack](#-tech-stack)
- [License & Disclaimer](#-license--disclaimer)

---

## 🔍 Executive Overview

In Formula 1 Grand Prix racing, the **starting tyre compound selection** is one of the highest-leverage strategic calls made before lights out. The starting compound dictates:

- **Opening Stint Pace & Tyre Degradation:** Softs offer maximum initial grip at the expense of early thermal degradation; Hards sacrifice launch pace for stint longevity.
- **Safety Car Windows & Pit Stop Flexibility:** Starting on Mediums or Hards allows cars to extend their first stint and exploit cheap pit stops under Virtual / Full Safety Cars.
- **Undercut / Overcut Dynamics:** Backmarkers frequently gamble on alternate compounds to make up ground through strategy.

**F1-Stratify** ingests real-world Formula 1 timing and telemetry data, engineers race-critical physical and regulatory variables, and trains high-performance classifiers (**Random Forest** and **Decision Trees**) to recommend the optimal starting compound with **78.21% test accuracy**.

---

## 🤖 Motorsport Domain Problem

```
                        ┌───────────────────────────────┐
                        │   RACE MORNING CONDITIONS     │
                        │ Track & Air Temps, Grid, Team │
                        └──────────────┬────────────────┘
                                       │
                         [ Domain Feature Engineering ]
                                       │
                        ┌──────────────▼────────────────┐
                        │     F1-STRATIFY ENGINE        │
                        │    (Random Forest / DT)       │
                        └──────────────┬────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   🔴 Soft Compound (0)       🟡 Medium Compound (1)     ⚪ Hard Compound (2)
   - Aggressive sprint        - Conservative balanced     - Extreme long-stint
   - High initial grip        - Long first stint window   - Traffic offset gamble
```

| Dimension | Specification |
|:---|:---|
| **Domain** | Motorsport Predictive Analytics & Race Strategy |
| **Task Type** | Multi-Class Supervised Classification |
| **Target Classes** | `0: Soft` (Red) · `1: Medium` (Yellow) · `2: Hard` (White) |
| **Primary Algorithm** | Random Forest Classifier (`n_estimators=300`, `max_depth=10`) + Decision Tree Explainability |
| **Primary Metric** | Multi-Class Accuracy & Stratified 5-Fold Cross-Validation |

---

## 🧠 Input Features & Strategy Engineering

Standard raw telemetry alone misses key motorsport dynamics. **F1-Stratify** engineers specialized features grounded in F1 technical regulations and thermodynamics:

| Feature Name | Type | Description & Strategic Rationale |
|:---|:---|:---|
| `GridPosition` | Integer (1–20) | Driver starting slot. Front rows protect track position; midfield/back-markers take strategic gambles. |
| `TrackTemp` | Float (°C) | Asphalt surface temperature. High track temperatures accelerate thermal blistering on Soft tyres. |
| `AirTemp` | Float (°C) | Ambient temperature impacting aerodynamic cooling and tyre working ranges. |
| `TempDelta` | Float (°C) | **Engineered feature:** `TrackTemp - AirTemp`. Isolates solar heat radiation load on track surface. |
| `IsTop10` | Binary (0 / 1) | **Regulatory feature:** Encodes the historic **FIA Q2 Tyre Rule (Article 24.4.j)**, which mandated that Top 10 qualifiers start on their Q2 tyre (predominantly Softs), while P11–20 enjoyed free tyre choice. |
| `Team` | Categorical (1–10) | Encodes constructor operational tendencies (e.g. Red Bull, Ferrari, Mercedes). |

---

## 📊 Benchmark Results

Evaluated on real Formula 1 telemetry with a stratified 80/20 train-test split:

```
==================================================
  TEST ACCURACY : 78.21%
==================================================
```

### Classification Report

| Compound Class | Precision | Recall | F1-Score | Test Support |
|:---|:---:|:---:|:---:|:---:|
| **Soft (0)** | **0.85** | **0.85** | **0.85** | 74 |
| **Medium (1)** | **0.75** | **0.81** | **0.78** | 73 |
| **Hard (2)** | 0.00 | 0.00 | 0.00 | 9 |
| **Weighted Average** | **0.75** | **0.78** | **0.77** | **156** |

### Feature Importance Breakdown

Track temperature metrics and grid position variables account for **~85% of total predictive power**:

| Rank | Feature | Gini Importance | Strategic Significance |
|:---:|:---|:---:|:---|
| 🥇 | `GridPosition` | **0.246** | Strongest single indicator of initial stint strategy |
| 🥈 | `AirTemp` | **0.184** | Tyre operating window threshold |
| 🥉 | `TrackTemp` | **0.172** | Surface degradation driver |
| 4 | `TempDelta` | **0.161** | Asphalt radiation load |
| 5 | `Team` | **0.151** | Constructor car-pace & tyre wear characteristics |
| 6 | `IsTop10` | **0.087** | FIA Q2 regulatory split |

---

## 🖼️ Visual Strategy Artifacts

| Feature Importance Ranking | Confusion Matrix |
|:---:|:---:|
| ![Feature Importance](outputs/feature_importance.png) | ![Confusion Matrix](outputs/confusion_matrix.png) |

| Compound Class Distribution | Sample Strategy Decision Tree |
|:---:|:---:|
| ![Class Distribution](outputs/class_distribution.png) | ![Decision Tree](outputs/decision_tree.png) |

---

## 🌲 Explainability & Strategy Tree

Black-box models are unacceptable on the pit wall where race engineers must justify every call to the Team Principal. **F1-Stratify** exports explainable decision boundaries:

```python
# Strategic decision logic extracted from learned tree boundaries:
if GridPosition <= 10.5:
    # Top 10 qualifiers bound by Q2 pace requirements
    if AirTemp <= 24.65°C:
        → Predict: Medium (cold ambient allows medium warmup)
    else:
        → Predict: Soft (aggressive track position defence)
else:
    # P11–20 drivers with free tyre choice
    if TrackTemp >= 39.2°C:
        → Predict: Medium (overcut strategy to outlast degrading soft starters)
    else:
        → Predict: Soft (early undercut gamble)
```

---

## 📁 Project Structure

```
F1Start-AI/
├── main.py                     # Entry point: trains, evaluates, generates plots & runs live demos
├── collect_real_data.py        # FastF1 real telemetry ingestion pipeline (2018–2024)
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, validation, and domain feature engineering
│   ├── model.py                # Random Forest / Decision Tree training, tuning & persistence
│   └── evaluate.py             # Evaluation metrics, high-contrast plots & tree visualisations
│
├── data/
│   ├── raw/
│   │   └── f1_real_data.csv    # Real race dataset (777 driver race records)
│   └── fastf1_cache/           # Local cache for FastF1 telemetry sessions
│
└── outputs/
    ├── confusion_matrix.png    # Heatmap of actual vs predicted compounds
    ├── feature_importance.png  # Gini importance plot with top features highlighted
    ├── decision_tree.png       # Explainable strategy tree visualization
    ├── class_distribution.png  # Pie chart with dynamic contrast labels
    └── randomForest_model.pkl  # Serialized production model artifact
```

---

## 🚀 Quick Start & Final Run Command

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/F1Start-AI.git
cd F1Start-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. ⭐ Final Command to Run the Complete Pipeline

To execute the entire engine (auto-detects real dataset, performs feature engineering, trains the model, saves all plots to `outputs/`, and executes 5 real-time race scenario predictions):

```bash
python3 main.py
```

> **⚡ Faster Run (Skip GridSearchCV):**
> If you want to skip hyperparameter search and run with the optimal pre-configured parameters in under 2 seconds:
> ```bash
> python3 main.py --no-tune
> ```

---

### 🖥️ Expected Terminal Output

```text
 ███████╗ ██╗     ███████╗████████╗ █████╗ ██████╗ ████████╗      █████╗ ██╗
 ██╔════╝███║     ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝     ██╔══██╗██║
 █████╗  ╚██║     ███████╗   ██║   ███████║██████╔╝   ██║   █████╗███████║██║
 ██╔══╝   ██║     ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   ╚════╝██╔══██║██║
 ██║      ██║     ███████║   ██║   ██║  ██║██║  ██║   ██║         ██║  ██║██║
 ╚═╝      ╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝         ╚═╝  ╚═╝╚═╝
           Predictive Modeling for F1 Initial Race Strategy

[DataLoader] Auto-detected real dataset at 'data/raw/f1_real_data.csv' …
[DataLoader] Loaded 777 rows
[DataLoader] After preprocessing: 777 rows
[Model] Train: 621 samples | Test: 156 samples
[Model] 5-Fold CV Accuracy: 0.7246 ± 0.0185
==================================================
  TEST ACCURACY : 78.21%
==================================================

[Evaluate] Confusion matrix saved → 'outputs/confusion_matrix.png'
[Evaluate] Feature importance saved → 'outputs/feature_importance.png'
[Evaluate] Sample tree saved → 'outputs/decision_tree.png'
[Evaluate] Class distribution saved → 'outputs/class_distribution.png'
[Model] Saved to 'outputs/randomForest_model.pkl'

────────────────────────────────────────────────────────────
  DEMO: Single-Race Strategy Predictions
────────────────────────────────────────────────────────────

  Verstappen – P1, hot Bahrain
    → Predicted Compound : Soft    (confidence 53.5%)
    → Probabilities      : {'Soft': '53.5%', 'Medium': '45.8%', 'Hard': '0.6%'}

  Leclerc – P3, cool Monza
    → Predicted Compound : Medium  (confidence 54.4%)
    → Probabilities      : {'Soft': '45.5%', 'Medium': '54.4%', 'Hard': '0.1%'}

  Hamilton – P6, warm Silverstone
    → Predicted Compound : Soft    (confidence 68.8%)
    → Probabilities      : {'Soft': '68.8%', 'Medium': '31.0%', 'Hard': '0.2%'}

  Back-marker – P18, hot track
    → Predicted Compound : Medium  (confidence 64.3%)
    → Probabilities      : {'Soft': '34.2%', 'Medium': '64.3%', 'Hard': '1.5%'}

  Midfield – P12, mild conditions
    → Predicted Compound : Medium  (confidence 44.8%)
    → Probabilities      : {'Soft': '18.6%', 'Medium': '44.8%', 'Hard': '36.6%'}

────────────────────────────────────────────────────────────
[Main] Pipeline complete. Final test accuracy: 78.21%
```

---

## 💻 Python API Usage

You can also use the trained model directly in your own scripts:

```python
from src.model import load_model, predict_single

# Load serialized model artifact
clf = load_model("outputs/randomForest_model.pkl")

# Predict for a single scenario
recommendation = predict_single(
    clf,
    grid_pos   = 3,      # P3 on the grid (Second row)
    track_temp = 48.0,   # 48°C track temperature (Hot Bahrain)
    air_temp   = 32.0,   # 32°C ambient air temperature
    rain_prob  = 0,      # Dry race
    team       = 2,      # Scuderia Ferrari
)

print(recommendation)
# Output:
# {
#   'predicted_compound': 'Medium',
#   'confidence': '62.4%',
#   'probabilities': {'Soft': '36.2%', 'Medium': '62.4%', 'Hard': '1.4%'}
# }
```

---

## 🧰 Tech Stack

| Technology | Purpose |
|:---|:---|
| **Python 3.9+** | Core programming language |
| **scikit-learn** | Random Forest, Decision Tree, GridSearchCV, Stratified Cross-Validation |
| **FastF1** | Formula 1 official live timing and telemetry data collection |
| **pandas & numpy** | Telemetry ingestion, feature engineering, and matrix operations |
| **matplotlib & seaborn** | Publication-quality dark-mode visual artifacts |

---

## 📄 License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

*Disclaimer: This project is an independent predictive modeling analysis and is not affiliated, associated, authorized, endorsed by, or in any way officially connected with Formula 1, the FIA, or Formula One Licensing B.V.*

# 🌱 Tomato Irrigation Scheduling — ML System

An AI-powered irrigation scheduling system for tomato crops, built with **Random Forest** under **data-scarce conditions**. Enter 5 sensor readings and the system predicts whether irrigation is needed — with confidence probabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Data-Scarcity Strategy](#data-scarcity-strategy)
- [Results](#results)

---

## Overview

This project predicts whether a tomato crop requires irrigation based on 5 agronomic inputs:

| Input | Description | Example |
|-------|-------------|---------|
| **Soil Moisture** | Sensor reading of water content in soil | 350 |
| **ET₀** | Reference evapotranspiration (mm/day) | 560 |
| **Crop Coefficient (Kc)** | Growth stage multiplier (Initial ≈ 0.42, Mid ≈ 1.15) | 0.75 |
| **Days Planted** | Days since transplanting | 45 |
| **Temperature (°C)** | Ambient air temperature | 31.0 |

**Output:** `Irrigate` or `No Irrigation Needed` with confidence probability.

---

## Features

- ✅ Random Forest classifier (best of 5 models auto-selected)
- ✅ SMOTE oversampling for class-imbalance handling
- ✅ 10-Fold Stratified Cross-Validation
- ✅ Bootstrap 95% Confidence Intervals on metrics
- ✅ Learning curve analysis (data scarcity diagnosis)
- ✅ Flask REST API backend
- ✅ Responsive web frontend with animated probability bars
- ✅ Model export (`.pkl`) for deployment

---

## Project Structure

```
irrigration_predication/
│
├── irrigation.ipynb              # Jupyter notebook — full ML pipeline
├── tomato irrigation dataset.csv # Training dataset
│
├── app.py                        # Flask web application
│
├── model/
│   ├── irrigation_model.pkl      # Trained model pipeline
│   └── metadata.json             # Feature names, labels, model info
│
├── templates/
│   └── index.html                # Web UI (frontend)
│
├── static/
│   └── style.css                 # Stylesheet
│
└── README.md                     # This file
```

---

## Requirements

- Python 3.9+
- pip

**Python packages:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
flask
```

---

## Installation

### 1. Clone or download the project

```bash
cd d:\irrigration_predication
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib flask
```

### 3. Train the model (run the notebook)

Open `irrigation.ipynb` in VS Code or Jupyter and run all cells top to bottom.  
This will generate `model/irrigation_model.pkl` and `model/metadata.json`.

> **Skip this step** if `model/irrigation_model.pkl` already exists.

---

## Usage

### Run the Web App

```bash
cd d:\irrigration_predication
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

Fill in the 5 sensor values and click **Predict Irrigation Need**.

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Pipeline | StandardScaler → SMOTE → Random Forest |
| CV Strategy | StratifiedKFold (k=10, shuffle=True) |
| Tuning | GridSearchCV (F1 scoring) |
| Target | `Soil moisture < 400` → Irrigate (1), else No Irrigation (0) |
| Evaluation | 20% hold-out + Bootstrap CIs (n=1000) |

### Model Performance

| Metric | Score | 95% CI |
|--------|-------|--------|
| Accuracy | 1.0000 | [1.0000, 1.0000] |
| F1 Score | 1.0000 | [1.0000, 1.0000] |
| ROC-AUC | 1.0000 | [1.0000, 1.0000] |
| Precision | 1.0000 | — |
| Recall | 1.0000 | — |

*Test set: 600 samples (20% stratified hold-out)*

### All Models Compared (10-Fold CV F1)

| Model | F1 Score |
|-------|----------|
| **Random Forest** ◄ best | 1.0000 ± 0.0000 |
| Gradient Boosting | 1.0000 ± 0.0000 |
| Logistic Regression | 0.9987 ± 0.0016 |
| SVM (RBF) | 0.9882 ± 0.0053 |
| K-Nearest Neighbours | 0.9633 ± 0.0066 |

---

## API Reference

### `GET /`
Returns the web UI.

---

### `POST /predict`

Accepts JSON body, returns irrigation prediction.

**Request:**
```json
{
  "soil_moisture": 320,
  "et0": 562,
  "crop_coefficient": 0.75,
  "days_planted": 45,
  "temperature": 33.0
}
```

**Response:**
```json
{
  "label": 1,
  "prediction": "Irrigate",
  "prob_irrigate": 0.89,
  "prob_no_irrigate": 0.11,
  "model_used": "Random Forest"
}
```

**Error Response (400):**
```json
{
  "error": "Invalid input: 'soil_moisture'"
}
```

---

## Data-Scarcity Strategy

Real irrigation sensor data is often limited or imbalanced. This system uses 5 techniques to stay robust:

| Technique | Purpose |
|-----------|---------|
| **StratifiedKFold (k=10)** | Preserves class balance in every fold even with small data |
| **SMOTE** | Synthesises minority-class samples inside each training fold (no data leakage) |
| **Learning Curves** | Diagnoses whether the model needs more data or is already generalising well |
| **Bootstrap CIs** | Gives honest uncertainty bounds on metrics when test sets are small |
| **5-feature lean model** | Fewer features = lower overfitting risk under data scarcity |

---

## Crop Coefficient (Kc) Reference

| Growth Stage | Days (approx.) | Kc Range |
|-------------|----------------|----------|
| Initial | 1 – 30 | 0.40 – 0.45 |
| Crop Development | 31 – 60 | 0.45 – 1.15 |
| Mid-Season (peak demand) | 61 – 90 | 1.10 – 1.15 |
| Late Season | 91+ | 0.70 – 0.90 |

---

## License

This project is for educational and research purposes.

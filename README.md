# 🛡️ Insurance Policy Lapse Predictor

> An end-to-end machine learning system for predicting insurance policy lapses, enabling proactive customer retention through AI-driven risk scoring and explainable predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Context](#-business-context)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Explainability](#-explainability-shap)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)

---

## 🔍 Overview

Policy lapse — when a customer stops paying insurance premiums — is one of the most costly problems in the life insurance industry. Acquiring a new customer costs 5–10× more than retaining an existing one, making early lapse detection a high-priority business problem.

This project builds a complete ML pipeline that:

- **Predicts** which policyholders are at risk of lapsing in the next period
- **Ranks** customers by lapse probability for targeted retention outreach  
- **Explains** each prediction using SHAP values, ensuring compliance-ready, auditable decisions
- **Deploys** as an interactive Streamlit dashboard supporting both single-policy analysis and bulk portfolio scoring

---

## 💼 Business Context

In the life insurance sector, **persistency** (the rate at which policies remain active) is a key performance indicator. A lapse prediction model enables the retention team to:

| Without ML | With ML |
|---|---|
| Reactive — contact customers after they lapse | Proactive — intervene before lapse occurs |
| Treat all customers equally | Prioritise high-risk customers for outreach |
| Manual, costly review process | Automated, ranked risk scoring |
| Black-box decisions | Explainable, auditable predictions |

The system outputs a **risk tier** (High / Medium / Low) and a **recommended action** for every policy, which can be fed directly into a CRM or retention workflow.

---

## 🎬 Demo

### Single Policy Analysis
Enter customer and policy details to get an instant lapse probability score, risk tier, and recommended retention action with key risk drivers explained in plain English.

### Batch Portfolio Scoring
Upload a CSV of your entire policy portfolio. The system scores every row, provides an aggregate risk breakdown, and exports a downloadable ranked retention list.

### Model Performance Dashboard
View ROC curves comparing all three models, confusion matrix at the optimal threshold, probability distribution plots, and SHAP feature importance.

---

## 📁 Project Structure

```
insurance-policy-lapse-predictor/
│
├── train_lapse_model.py       # Full ML training pipeline
│   ├── Data loading & renaming to insurance terminology
│   ├── Feature engineering (5 derived features)
│   ├── Preprocessing pipeline (StandardScaler + OrdinalEncoder)
│   ├── 3-model comparison with 5-fold cross-validation
│   ├── Probability calibration (Platt scaling)
│   ├── Threshold tuning optimised for Recall ≥ 0.75
│   ├── ROC curves, confusion matrix, distribution plots
│   └── SHAP explainability plots
│
├── lapse_app.py               # Streamlit web application
│   ├── Tab 1: Single policy risk scorer
│   ├── Tab 2: Batch CSV portfolio scoring
│   └── Tab 3: Model performance & SHAP dashboard
│
├── telco_churn.csv            # Dataset (IBM Telco, remapped to insurance)
├── requirements.txt           # Python dependencies
│
├── best_lapse_model.pkl       # Saved model pipeline [generated on train]
├── model_metadata.pkl         # Model metrics & feature names [generated on train]
├── training_results.png       # ROC curves & confusion matrix [generated on train]
└── shap_summary.png           # SHAP feature importance plot [generated on train]
```

> **Note:** The `.pkl` and `.png` files are generated automatically when you run `train_lapse_model.py`. They are not included in the repository.

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing

- **Categorical features** encoded using `OrdinalEncoder` with unknown-value handling
- **Numeric features** standardised using `StandardScaler`
- Both transformers wrapped in a `ColumnTransformer` inside a `sklearn.Pipeline` for clean train/test separation and zero data leakage

### 2. Handling Class Imbalance

The dataset has a **14.4% lapse rate** — a meaningful class imbalance. Two techniques are applied:

- **Class weights** set to `{Non-lapsed: 1, Lapsed: 2}` — penalises the model more for missing lapse cases without over-correcting
- **Probability calibration** via `CalibratedClassifierCV` (Platt scaling) — ensures predicted probabilities are well-calibrated and reflect true lapse likelihood rather than raw decision scores

### 3. Model Comparison

Three models are trained and compared using **5-fold stratified cross-validation**:

| Model | CV AUC-ROC | Std Dev |
|---|---|---|
| **Logistic Regression** ✅ | **0.7805** | ±0.0259 |
| Random Forest | 0.7753 | ±0.0207 |
| Gradient Boosting | 0.7710 | ±0.0187 |

Logistic Regression won — a common outcome on structured tabular data where feature-target relationships are approximately linear, and where strong regularisation prevents overfitting on a moderately sized dataset.

### 4. Threshold Tuning

The default classification threshold (0.5) is not optimal for imbalanced problems. The threshold is tuned on the precision-recall curve to find the point where **Recall ≥ 0.75** at maximum precision:

- **Optimal threshold:** `0.165`
- **At this threshold:** Precision = 0.273, Recall = 0.754
- This means the model **catches 75% of all actual lapses**, at the cost of some false alarms — an acceptable trade-off since the cost of missing a lapse far exceeds the cost of an unnecessary retention call

### 5. Risk Tier Classification

Predicted probabilities are mapped to risk tiers based on the dataset's actual probability distribution:

| Risk Tier | Probability Range | Action |
|---|---|---|
| 🔴 High | ≥ 25% (top 20% of customers) | Immediate senior agent outreach |
| 🟡 Medium | 13% – 25% (next 30%) | Schedule follow-up call within 7 days |
| 🟢 Low | < 13% (bottom 50%) | Periodic monitoring only |

---

## 📊 Model Performance

**Test Set Results (Logistic Regression)**

| Metric | Value |
|---|---|
| AUC-ROC | **0.7825** |
| Average Precision | 0.3314 |
| Recall (Lapsed class) | 0.754 |
| Decision Threshold | 0.165 |
| Test Set Size | 1,409 policies |

> **Why AUC-ROC over accuracy?** A naive model that predicts "no lapse" for every customer would achieve 85.6% accuracy on this dataset — yet be completely useless. AUC-ROC measures discriminative ability across all thresholds regardless of class distribution, making it the correct evaluation metric here.

---

## 🔧 Feature Engineering

Five derived features are created beyond the raw dataset columns:

| Feature | Formula | Business Meaning |
|---|---|---|
| `avg_monthly_premium` | `total_paid / (tenure + 1)` | Historical average payment amount |
| `premium_consistency` | `monthly_premium / avg_monthly_premium` | How stable the customer's payments are |
| `high_premium_flag` | `1 if premium > median else 0` | Binary flag for affordability pressure |
| `tenure_band` | Bucketed: 0-1yr, 1-2yr, 2-4yr, 4+yr | Non-linear tenure effect on lapse risk |

---

## 🔬 Explainability (SHAP)

All predictions are backed by **SHAP (SHapley Additive exPlanations)** values, which decompose each prediction into per-feature contributions. This is critical in regulated industries where decisions must be auditable.

Key drivers of lapse risk identified by the model:

- **Policy contract type** — Annual Renewable policies lapse at 2.5× the rate of 10-Year Term
- **Policy tenure** — Lapse risk is highest in the first 12 months and declines with tenure
- **Payment method** — Electronic check is strongly associated with higher lapse rates
- **Senior citizen status** — Elevated lapse tendency in the 60+ segment
- **Premium consistency** — Irregular payment history is a leading indicator of future lapse

The Streamlit app also surfaces plain-English risk drivers for each individual prediction, meeting IRDAI-style explainability requirements without requiring the user to interpret SHAP plots directly.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/insurance-policy-lapse-predictor.git
cd insurance-policy-lapse-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the model
```bash
python train_lapse_model.py
```
This will:
- Load and preprocess `telco_churn.csv`
- Train and compare 3 models with 5-fold cross-validation
- Save `best_lapse_model.pkl` and `model_metadata.pkl`
- Generate `training_results.png` and `shap_summary.png`
- Print a full training report to the console

Expected runtime: ~60–90 seconds

### Step 2 — Launch the app
```bash
streamlit run lapse_app.py
```
The app will open automatically at `http://localhost:8501`

### Batch Scoring
Upload any CSV to the **Batch Scoring** tab. The app automatically detects and handles:
- IBM Telco Churn format
- Bank Customer Churn format (Kaggle)
- Generic CSVs (missing columns filled with conservative defaults)

---

## 📂 Dataset

**Source:** IBM Telco Customer Churn Dataset, remapped to insurance domain terminology

| Original Column | Insurance Equivalent | Description |
|---|---|---|
| `tenure` | `policy_tenure_months` | Months since policy inception |
| `Contract` | `policy_contract_type` | Annual Renewable / 5-Year / 10-Year |
| `MonthlyCharges` | `monthly_premium` | Monthly premium amount |
| `TotalCharges` | `total_premium_paid` | Cumulative premiums paid |
| `InternetService` | `investment_plan_type` | ULIP / Endowment / Term Only |
| `OnlineSecurity` | `has_accident_cover` | Accidental death cover |
| `TechSupport` | `has_critical_illness_cover` | Critical illness rider |
| `Churn` | `policy_lapsed` | Target variable |

**Dataset stats:**
- 7,043 policies
- 14.4% lapse rate
- 80/20 train/test split with stratification

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | scikit-learn 1.3+ |
| Explainability | SHAP |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Serialisation | Pickle |

---

## 💡 Key Learnings

- **Simpler models can outperform complex ones** on structured tabular data — Logistic Regression beat Random Forest and Gradient Boosting here because the feature-target relationships are approximately linear and the dataset is moderately sized
- **Accuracy is a misleading metric on imbalanced data** — AUC-ROC and recall on the minority class are the correct measures for lapse prediction
- **Probability calibration matters for production** — raw model scores from tree-based models can be miscalibrated; Platt scaling corrects this to ensure probabilities reflect true likelihood
- **Explainability is not optional in insurance** — SHAP values make predictions auditable for regulatory compliance and give the retention team actionable insight into why a specific customer is flagged
- **Threshold tuning is a business decision** — choosing 0.165 over the default 0.5 was driven by the asymmetric cost of false negatives (missed lapses) vs. false positives (unnecessary outreach)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Satvik Asthana**  
B.Tech Robotics & AI — Manav Rachna University  
[LinkedIn](https://linkedin.com/in/satvik-asthana-835898256) · [GitHub](https://github.com/SatvikAsthana) · asthanasatvik21@gmail.com

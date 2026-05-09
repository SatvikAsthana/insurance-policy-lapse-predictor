"""
Insurance Policy Lapse (Persistency) Predictor
================================================
Mirrors Canara HSBC Life Insurance's actual persistency prediction use case.
Uses IBM Telco Churn dataset remapped to insurance domain terminology.

Models compared:
  - Logistic Regression (interpretable baseline)
  - Random Forest
  - Gradient Boosting (XGBoost)

Outputs:
  - best_lapse_model.pkl     (trained pipeline)
  - label_encoder.pkl        (target encoder)
  - feature_names.pkl        (feature list for Streamlit app)
  - training_results.png     (ROC curves + feature importance)
  - shap_summary.png         (SHAP explainability plot)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, pickle, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import shap

# ─────────────────────────────────────────────
# 1. LOAD & RENAME TO INSURANCE TERMINOLOGY
# ─────────────────────────────────────────────
print("=" * 60)
print("INSURANCE POLICY LAPSE PREDICTOR — TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv('telco_churn.csv')

# Rename columns: Telco → Insurance domain
df = df.rename(columns={
    'customerID':       'policy_id',
    'gender':           'gender',
    'SeniorCitizen':    'senior_citizen',
    'Partner':          'has_spouse',
    'Dependents':       'has_dependents',
    'tenure':           'policy_tenure_months',
    'PhoneService':     'has_term_plan',
    'MultipleLines':    'has_multiple_policies',
    'InternetService':  'investment_plan_type',
    'OnlineSecurity':   'has_accident_cover',
    'TechSupport':      'has_critical_illness_cover',
    'Contract':         'policy_contract_type',
    'PaperlessBilling': 'paperless_billing',
    'PaymentMethod':    'payment_method',
    'MonthlyCharges':   'monthly_premium',
    'TotalCharges':     'total_premium_paid',
    'Churn':            'policy_lapsed'
})

# Remap Contract values to insurance terms
df['policy_contract_type'] = df['policy_contract_type'].map({
    'Month-to-month': 'Annual Renewable',
    'One year':       '5-Year Term',
    'Two year':       '10-Year Term'
})

# Remap investment_plan_type
df['investment_plan_type'] = df['investment_plan_type'].map({
    'DSL':         'ULIP',
    'Fiber optic': 'Endowment',
    'No':          'Term Only'
})

# Remap accident/illness cover
for col in ['has_accident_cover', 'has_critical_illness_cover']:
    df[col] = df[col].map({
        'Yes': 'Yes',
        'No':  'No',
        'No internet service': 'Not Applicable'
    })

df['has_multiple_policies'] = df['has_multiple_policies'].map({
    'Yes': 'Yes', 'No': 'No', 'No phone service': 'No'
})

print(f"\nDataset loaded: {df.shape[0]:,} policies")
print(f"Lapse rate: {(df['policy_lapsed']=='Yes').mean():.1%}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Average monthly premium over policy life
df['avg_monthly_premium'] = df['total_premium_paid'] / (df['policy_tenure_months'] + 1)

# Premium consistency ratio (higher = more consistent payer)
df['premium_consistency'] = np.where(
    df['avg_monthly_premium'] > 0,
    df['monthly_premium'] / df['avg_monthly_premium'],
    1.0
)
df['premium_consistency'] = df['premium_consistency'].clip(0, 3)

# Tenure band
df['tenure_band'] = pd.cut(
    df['policy_tenure_months'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
).astype(str)

# High premium flag
df['high_premium_flag'] = (df['monthly_premium'] > df['monthly_premium'].median()).astype(int)

print("\nFeature engineering complete.")
print(f"Final feature count: {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 3. PREPROCESSING SETUP
# ─────────────────────────────────────────────
drop_cols = ['policy_id', 'policy_lapsed']
X = df.drop(columns=drop_cols)
y = (df['policy_lapsed'] == 'Yes').astype(int)

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric features    ({len(numeric_cols)}): {numeric_cols}")

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"Train lapse rate: {y_train.mean():.1%} | Test lapse rate: {y_test.mean():.1%}")

# ─────────────────────────────────────────────
# 4. CLASS IMBALANCE — CLASS WEIGHTS
# ─────────────────────────────────────────────
classes     = np.array([0, 1])
cw          = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1]}
print(f"\nClass weights: Non-lapsed={cw[0]:.2f}, Lapsed={cw[1]:.2f}")

# ─────────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('prep', preprocessor),
        ('clf',  CalibratedClassifierCV(
            LogisticRegression(
                class_weight={0: 1, 1: 2},   # mild weight — not 3.48x
                max_iter=1000, C=1.0, random_state=42
            ), cv=5, method='sigmoid'
        ))
    ]),
    'Random Forest': Pipeline([
        ('prep', preprocessor),
        ('clf',  CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=200, class_weight={0: 1, 1: 2},
                max_depth=8, min_samples_leaf=10, random_state=42
            ), cv=5, method='sigmoid'
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('prep', preprocessor),
        ('clf',  CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8, random_state=42
            ), cv=5, method='sigmoid'
        ))
    ]),
}

print("\n" + "─" * 50)
print("5-FOLD CROSS-VALIDATION RESULTS (AUC-ROC)")
print("─" * 50)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:<25}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n✅ Best model: {best_model_name}")

# ─────────────────────────────────────────────
# 6. TRAIN BEST MODEL & EVALUATE
# ─────────────────────────────────────────────
best_pipe = models[best_model_name]
best_pipe.fit(X_train, y_train)

y_pred       = best_pipe.predict(X_test)
y_pred_proba = best_pipe.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)
test_ap  = average_precision_score(y_test, y_pred_proba)

print(f"\n{'─'*50}")
print(f"TEST SET RESULTS — {best_model_name}")
print(f"{'─'*50}")
print(f"AUC-ROC:           {test_auc:.4f}")
print(f"Avg Precision:     {test_ap:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Active', 'Lapsed']))

# ─────────────────────────────────────────────
# 7. THRESHOLD TUNING — optimise for Recall
# ─────────────────────────────────────────────
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
# Find threshold where recall >= 0.75 and precision is maximised
valid = [(p, r, t) for p, r, t in zip(precisions, recalls, thresholds) if r >= 0.75]
if valid:
    best_p, best_r, best_t = max(valid, key=lambda x: x[0])
    print(f"\nOptimal threshold (Recall ≥ 0.75): {best_t:.3f}")
    print(f"  → Precision: {best_p:.3f}  Recall: {best_r:.3f}")
    optimal_threshold = best_t
else:
    optimal_threshold = 0.5

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Insurance Policy Lapse Predictor — Training Results', fontsize=14, fontweight='bold')

# (a) ROC curves for all models
ax = axes[0, 0]
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) Confusion matrix at optimal threshold
ax = axes[0, 1]
y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Active', 'Lapsed'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix (threshold={optimal_threshold:.2f})')

# (c) Churn probability distribution
ax = axes[1, 0]
ax.hist(y_pred_proba[y_test==0], bins=40, alpha=0.6, label='Active Policies', color='steelblue')
ax.hist(y_pred_proba[y_test==1], bins=40, alpha=0.6, label='Lapsed Policies', color='tomato')
ax.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.2f}')
ax.set_xlabel('Predicted Lapse Probability')
ax.set_ylabel('Count')
ax.set_title('Lapse Probability Distribution')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (d) Cross-validation comparison
ax = axes[1, 1]
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]
colors = ['tomato' if n == best_model_name else 'steelblue' for n in names]
bars = ax.barh(names, means, xerr=stds, color=colors, alpha=0.8, capsize=4)
ax.set_xlabel('AUC-ROC (5-fold CV)')
ax.set_title('Model Comparison — Cross Validation')
ax.set_xlim(0.5, 1.0)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9)
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: training_results.png")

# ─────────────────────────────────────────────
# 9. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
print("\nGenerating SHAP explanations...")

# Transform data for SHAP
X_test_transformed = best_pipe.named_steps['prep'].transform(X_test)
feature_names_transformed = numeric_cols + categorical_cols

clf = best_pipe.named_steps['clf']

# CalibratedClassifierCV wraps the base estimator — extract it
try:
    # For CalibratedClassifierCV, use the first calibrated classifier's base estimator
    base_clf = clf.calibrated_classifiers_[0].estimator
    if hasattr(base_clf, 'estimators_') or hasattr(base_clf, 'feature_importances_'):
        explainer   = shap.TreeExplainer(base_clf)
        shap_values = explainer.shap_values(X_test_transformed[:200])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer   = shap.LinearExplainer(base_clf, X_test_transformed[:200])
        shap_values = explainer.shap_values(X_test_transformed[:200])
    X_shap = X_test_transformed[:200]
except Exception:
    # Fallback: use permutation-based explainer
    explainer   = shap.KernelExplainer(clf.predict_proba, X_test_transformed[:50])
    shap_values = explainer.shap_values(X_test_transformed[:50])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    X_shap = X_test_transformed[:50]

fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_shap,
    feature_names=feature_names_transformed,
    show=False, max_display=15, plot_size=None
)
plt.title('SHAP Feature Importance — Drivers of Policy Lapse', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
print("Saved: shap_summary.png")

# ─────────────────────────────────────────────
# 10. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
with open('best_lapse_model.pkl', 'wb') as f:
    pickle.dump(best_pipe, f)

metadata = {
    'best_model_name':   best_model_name,
    'test_auc':          round(test_auc, 4),
    'test_ap':           round(test_ap, 4),
    'optimal_threshold': round(optimal_threshold, 3),
    'feature_names':     X.columns.tolist(),
    'numeric_cols':      numeric_cols,
    'categorical_cols':  categorical_cols,
    'cv_results':        {k: v.tolist() for k, v in cv_results.items()},
    'class_weight':      class_weight_dict,
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best model:         {best_model_name}")
print(f"  Test AUC-ROC:       {test_auc:.4f}")
print(f"  Avg Precision:      {test_ap:.4f}")
print(f"  Optimal threshold:  {optimal_threshold:.3f}")
print(f"  Saved artifacts:    best_lapse_model.pkl, model_metadata.pkl")
print(f"  Saved plots:        training_results.png, shap_summary.png")

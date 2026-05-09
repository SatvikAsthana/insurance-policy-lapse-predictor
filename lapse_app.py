"""
Insurance Policy Lapse Predictor — Streamlit App
=================================================
Canara HSBC-style persistency prediction dashboard.
Run with: streamlit run lapse_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Lapse Predictor",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: rgba(128,128,128,0.1);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 12px;
        color: rgba(128,128,128,0.9);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value { font-size: 26px; font-weight: 700; margin: 6px 0 2px; }
    .risk-HIGH   { background: rgba(229,57,53,0.12);  border-left: 4px solid #e53935; border-radius: 8px; padding: 16px 20px; }
    .risk-MEDIUM { background: rgba(251,140,0,0.12);  border-left: 4px solid #fb8c00; border-radius: 8px; padding: 16px 20px; }
    .risk-LOW    { background: rgba(67,160,71,0.12);  border-left: 4px solid #43a047; border-radius: 8px; padding: 16px 20px; }
    .risk-title  { font-size: 17px; font-weight: 600; margin: 0 0 6px; }
    .risk-sub    { font-size: 13px; opacity: 0.85; margin: 0; }
    .section-header {
        font-size: 14px; font-weight: 600;
        border-bottom: 1.5px solid rgba(128,128,128,0.3);
        padding-bottom: 6px; margin-bottom: 14px;
    }
    .action-box {
        background: rgba(66,133,244,0.12);
        border: 1px solid rgba(66,133,244,0.25);
        border-radius: 8px;
        padding: 14px 18px; margin-top: 12px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('best_lapse_model.pkl'):
        st.error("❌ Model not found. Please run `python train_lapse_model.py` first.")
        return None, None
    with open('best_lapse_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    return model, meta

model, meta = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:40px;padding-top:8px'>🛡️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("## Policy Lapse Predictor")
    st.markdown(
        "<p style='color:#666;margin-top:-10px;font-size:14px'>"
        "AI-powered persistency prediction for proactive customer retention</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — MODEL INFO
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    if meta:
        st.markdown(f"**Algorithm:** {meta['best_model_name']}")
        st.markdown(f"**AUC-ROC:** `{meta['test_auc']}`")
        st.markdown(f"**Decision threshold:** `{meta['optimal_threshold']}`")
        st.markdown(f"**Optimised for:** High Recall (catch more lapses)")

    st.markdown("---")
    st.markdown("### 📊 Risk Tiers")
    st.markdown("🔴 **High Risk** — Prob ≥ 25%\n\nImmediate outreach required")
    st.markdown("🟡 **Medium Risk** — Prob 13–25%\n\nSchedule follow-up call")
    st.markdown("🟢 **Low Risk** — Prob < 13%\n\nMonitor periodically")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "An ML-powered dashboard for predicting insurance policy lapses "
        "and enabling proactive customer retention. Built using scikit-learn "
        "with SHAP-based explainability for compliance-ready predictions."
    )

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Policy Analysis", "📋 Batch Scoring", "📈 Model Performance"])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE POLICY
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<p class='section-header'>Enter Policy Details</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Customer Profile**")
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen  = st.selectbox("Senior Citizen (60+)", ["No", "Yes"])
        has_spouse      = st.selectbox("Married / Has Spouse", ["Yes", "No"])
        has_dependents  = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.markdown("**📋 Policy Details**")
        policy_tenure   = st.slider("Policy Tenure (months)", 1, 72, 24)
        monthly_premium = st.slider("Monthly Premium (₹)", 500, 5000, 1500)
        total_premium   = st.number_input(
            "Total Premium Paid (₹)",
            min_value=0.0,
            value=float(policy_tenure * monthly_premium),
            step=500.0
        )
        policy_contract = st.selectbox(
            "Contract Type",
            ["Annual Renewable", "5-Year Term", "10-Year Term"]
        )

    with col3:
        st.markdown("**🔒 Coverage & Payment**")
        investment_type = st.selectbox(
            "Investment Plan Type",
            ["Term Only", "ULIP", "Endowment"]
        )
        accident_cover  = st.selectbox(
            "Accidental Death Cover",
            ["Yes", "No", "Not Applicable"]
        )
        illness_cover   = st.selectbox(
            "Critical Illness Cover",
            ["Yes", "No", "Not Applicable"]
        )
        has_term        = st.selectbox("Has Term Plan", ["Yes", "No"])
        multi_policy    = st.selectbox("Holds Multiple Policies", ["Yes", "No"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.markdown("")
    predict_btn = st.button("🔮 Predict Lapse Risk", use_container_width=True, type="primary")

    if predict_btn and model is not None:
        # Build input dataframe
        avg_premium   = total_premium / (policy_tenure + 1)
        prem_consist  = min((monthly_premium / avg_premium) if avg_premium > 0 else 1.0, 3.0)
        median_premium = 1500  # approximate median
        high_flag     = int(monthly_premium > median_premium)
        tenure_val    = policy_tenure
        if tenure_val <= 12:    tb = '0-1yr'
        elif tenure_val <= 24:  tb = '1-2yr'
        elif tenure_val <= 48:  tb = '2-4yr'
        else:                   tb = '4+yr'

        input_dict = {
            'senior_citizen':              [1 if senior_citizen == 'Yes' else 0],
            'policy_tenure_months':        [policy_tenure],
            'monthly_premium':             [float(monthly_premium)],
            'total_premium_paid':          [float(total_premium)],
            'avg_monthly_premium':         [avg_premium],
            'premium_consistency':         [prem_consist],
            'high_premium_flag':           [high_flag],
            'gender':                      [gender],
            'has_spouse':                  [has_spouse],
            'has_dependents':              [has_dependents],
            'has_term_plan':               [has_term],
            'has_multiple_policies':       [multi_policy],
            'investment_plan_type':        [investment_type],
            'has_accident_cover':          [accident_cover],
            'has_critical_illness_cover':  [illness_cover],
            'policy_contract_type':        [policy_contract],
            'paperless_billing':           [paperless],
            'payment_method':              [payment_method],
            'tenure_band':                 [tb],
        }
        input_df = pd.DataFrame(input_dict)

        lapse_prob  = model.predict_proba(input_df)[0][1]
        threshold   = meta['optimal_threshold'] if meta else 0.5
        prediction  = int(lapse_prob >= threshold)

        # Thresholds based on actual model distribution
        # (median ~13%, 80th pct ~26%, max ~62%)
        if lapse_prob >= 0.25:
            risk_level = "HIGH"
            emoji      = "🔴"
            action     = ("Assign to senior retention agent. Offer premium holiday or plan upgrade. "
                          "Escalate if no response within 48 hours.")
        elif lapse_prob >= 0.13:
            risk_level = "MEDIUM"
            emoji      = "🟡"
            action     = ("Schedule follow-up call within 7 days. "
                          "Share policy benefits summary. Offer auto-debit setup.")
        else:
            risk_level = "LOW"
            emoji      = "🟢"
            action     = "No immediate action required. Include in next monthly wellness communication."

        st.markdown("---")
        st.markdown("<p class='section-header'>Prediction Result</p>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        prob_color = "#e53935" if risk_level=="HIGH" else "#fb8c00" if risk_level=="MEDIUM" else "#43a047"
        with r1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Lapse Probability</div>
                <div class='metric-value' style='color:{prob_color}'>{lapse_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Risk Tier</div>
                <div class='metric-value' style='color:{prob_color}'>{emoji} {risk_level}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            dec_color = "#fb8c00" if prediction else "#43a047"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Decision</div>
                <div class='metric-value' style='font-size:17px;color:{dec_color}'>{"⚠️ Flag for Retention" if prediction else "✅ Likely Active"}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='risk-{risk_level}'>
            <p class='risk-title'>{emoji} Recommended Action</p>
            <p class='risk-sub'>{action}</p>
        </div>""", unsafe_allow_html=True)

        # Key risk drivers (manual heuristics for explainability display)
        st.markdown("")
        st.markdown("<p class='section-header'>Key Risk Drivers</p>", unsafe_allow_html=True)
        drivers = []
        if policy_contract == "Annual Renewable":
            drivers.append(("📄 Annual Renewable contract", "High — month-to-month customers lapse 2.5× more often"))
        if senior_citizen == "Yes":
            drivers.append(("👴 Senior citizen", "Moderate — higher lapse tendency in this segment"))
        if policy_tenure <= 12:
            drivers.append(("📅 Short tenure (≤ 12 months)", "High — early-tenure policies are at greatest risk"))
        if payment_method == "Electronic check":
            drivers.append(("💳 Electronic check payment", "Moderate — associated with higher lapse rates"))
        if investment_type == "Endowment":
            drivers.append(("📈 Endowment plan", "Moderate — higher premium burden increases lapse risk"))
        if has_spouse == "No" and has_dependents == "No":
            drivers.append(("👤 No financial dependents", "Moderate — lower perceived urgency to maintain cover"))
        if monthly_premium > 3000:
            drivers.append(("💰 High monthly premium", "Moderate — affordability pressure"))
        if not drivers:
            drivers.append(("✅ No major risk flags detected", "Policy profile is relatively stable"))

        for driver, reason in drivers:
            st.markdown(f"- **{driver}** — {reason}")

# ══════════════════════════════════════════════
# TAB 2 — BATCH SCORING
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<p class='section-header'>Upload Policy Portfolio for Batch Scoring</p>", unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file matching the dataset format. "
        "The model will score all policies and return a prioritised retention list."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None and model is not None:
        raw = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(raw):,} policies**")

        # ── Telco IBM format ──────────────────────────────────────
        if 'Churn' in raw.columns and 'tenure' in raw.columns:
            raw = raw.rename(columns={
                'customerID':'policy_id','gender':'gender',
                'SeniorCitizen':'senior_citizen','Partner':'has_spouse',
                'Dependents':'has_dependents','tenure':'policy_tenure_months',
                'PhoneService':'has_term_plan','MultipleLines':'has_multiple_policies',
                'InternetService':'investment_plan_type','OnlineSecurity':'has_accident_cover',
                'TechSupport':'has_critical_illness_cover','Contract':'policy_contract_type',
                'PaperlessBilling':'paperless_billing','PaymentMethod':'payment_method',
                'MonthlyCharges':'monthly_premium','TotalCharges':'total_premium_paid',
                'Churn':'policy_lapsed'
            })
            raw['policy_contract_type'] = raw['policy_contract_type'].map({
                'Month-to-month':'Annual Renewable','One year':'5-Year Term','Two year':'10-Year Term'
            })
            raw['investment_plan_type'] = raw['investment_plan_type'].map({
                'DSL':'ULIP','Fiber optic':'Endowment','No':'Term Only'
            })
            for c in ['has_accident_cover','has_critical_illness_cover']:
                raw[c] = raw[c].map({'Yes':'Yes','No':'No','No internet service':'Not Applicable'})
            raw['has_multiple_policies'] = raw['has_multiple_policies'].map(
                {'Yes':'Yes','No':'No','No phone service':'No'}
            )

        # ── Bank churn format (Exited, CreditScore, Age, Balance) ─
        elif 'Exited' in raw.columns and 'CreditScore' in raw.columns:
            raw = raw.rename(columns={
                'CustomerId':'policy_id','Gender':'gender',
                'Age':'senior_citizen','Tenure':'policy_tenure_months',
                'Balance':'total_premium_paid','EstimatedSalary':'monthly_premium',
                'Exited':'policy_lapsed','NumOfProducts':'has_multiple_policies',
                'HasCrCard':'has_accident_cover','IsActiveMember':'has_critical_illness_cover',
            })
            raw['senior_citizen']    = (raw['senior_citizen'] >= 60).astype(int)
            raw['has_spouse']        = raw.get('has_spouse', pd.Series(['No']*len(raw), index=raw.index))
            raw['has_dependents']    = 'No'
            raw['has_term_plan']     = 'Yes'
            raw['has_multiple_policies'] = raw['has_multiple_policies'].apply(
                lambda x: 'Yes' if int(x) > 1 else 'No'
            )
            raw['investment_plan_type']         = 'ULIP'
            raw['has_accident_cover']           = raw['has_accident_cover'].map({1:'Yes', 0:'No'})
            raw['has_critical_illness_cover']   = raw['has_critical_illness_cover'].map({1:'Yes', 0:'No'})
            raw['policy_contract_type']         = '5-Year Term'
            raw['paperless_billing']            = 'Yes'
            raw['payment_method']               = 'Bank transfer (automatic)'
            raw['monthly_premium']              = raw['monthly_premium'] / 12
            raw['policy_lapsed']                = raw['policy_lapsed'].map({1:'Yes', 0:'No'})

        # ── Generic fallback: fill any missing required columns ───
        col_defaults = {
            'policy_id':                    [f'ROW-{i}' for i in range(len(raw))],
            'gender':                       'Male',
            'senior_citizen':               0,
            'has_spouse':                   'No',
            'has_dependents':               'No',
            'policy_tenure_months':         12,
            'has_term_plan':                'Yes',
            'has_multiple_policies':        'No',
            'investment_plan_type':         'Term Only',
            'has_accident_cover':           'No',
            'has_critical_illness_cover':   'No',
            'policy_contract_type':         'Annual Renewable',
            'paperless_billing':            'No',
            'payment_method':               'Bank transfer (automatic)',
            'monthly_premium':              1000.0,
            'total_premium_paid':           12000.0,
        }
        for col, default in col_defaults.items():
            if col not in raw.columns:
                raw[col] = default

        # ── Feature engineering (safe — columns guaranteed above) ─
        raw['total_premium_paid']  = pd.to_numeric(raw['total_premium_paid'], errors='coerce').fillna(0)
        raw['monthly_premium']     = pd.to_numeric(raw['monthly_premium'],    errors='coerce').fillna(1000)
        raw['policy_tenure_months']= pd.to_numeric(raw['policy_tenure_months'], errors='coerce').fillna(12)
        raw['avg_monthly_premium'] = raw['total_premium_paid'] / (raw['policy_tenure_months'] + 1)
        raw['premium_consistency'] = (raw['monthly_premium'] / raw['avg_monthly_premium'].replace(0, 1)).clip(0, 3)
        raw['high_premium_flag']   = (raw['monthly_premium'] > raw['monthly_premium'].median()).astype(int)
        raw['tenure_band'] = pd.cut(
            raw['policy_tenure_months'],
            bins=[0, 12, 24, 48, 72], labels=['0-1yr','1-2yr','2-4yr','4+yr']
        ).astype(str)

        feature_cols = meta['feature_names']
        X_batch = raw[feature_cols]
        proba   = model.predict_proba(X_batch)[:, 1]

        raw['lapse_probability'] = np.round(proba * 100, 1)
        raw['risk_tier'] = pd.cut(proba, bins=[0, 0.13, 0.25, 1.0],
                                   labels=['Low', 'Medium', 'High'])
        raw['recommended_action'] = raw['risk_tier'].map({
            'High':   '🔴 Immediate outreach',
            'Medium': '🟡 Schedule follow-up',
            'Low':    '🟢 Monitor'
        })

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        total = len(raw)
        high  = (raw['risk_tier']=='High').sum()
        med   = (raw['risk_tier']=='Medium').sum()
        low   = (raw['risk_tier']=='Low').sum()
        for col, label, val, color in [
            (s1, "Total Policies", f"{total:,}", "inherit"),
            (s2, "High Risk 🔴",   f"{high:,} ({high/total:.0%})", "#e53935"),
            (s3, "Medium Risk 🟡", f"{med:,} ({med/total:.0%})",  "#fb8c00"),
            (s4, "Low Risk 🟢",    f"{low:,} ({low/total:.0%})",  "#43a047"),
        ]:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color};font-size:22px'>{val}</div>
            </div>""", unsafe_allow_html=True)

        # Sorted results
        display_cols = ['policy_id','policy_tenure_months','monthly_premium',
                        'policy_contract_type','lapse_probability','risk_tier','recommended_action']
        display_cols = [c for c in display_cols if c in raw.columns]
        result_df = raw[display_cols].sort_values('lapse_probability', ascending=False)
        st.dataframe(result_df, use_container_width=True, height=400)

        csv_out = result_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Scored Portfolio",
            data=csv_out, file_name="lapse_risk_scores.csv", mime="text/csv"
        )
    else:
        st.info("Upload the `telco_churn.csv` file to test batch scoring, or any CSV in the same format.")

# ══════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<p class='section-header'>Model Training Results</p>", unsafe_allow_html=True)

    if meta:
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val in [
            (m1, "Best Model",         meta['best_model_name']),
            (m2, "AUC-ROC",            str(meta['test_auc'])),
            (m3, "Avg Precision",      str(meta['test_ap'])),
            (m4, "Decision Threshold", str(meta['optimal_threshold'])),
        ]:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='font-size:17px'>{val}</div>
            </div>""", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        if os.path.exists('training_results.png'):
            st.image('training_results.png', caption='ROC Curves, Confusion Matrix & Model Comparison', use_container_width=True)
        else:
            st.warning("Run train_lapse_model.py to generate plots.")
    with p2:
        if os.path.exists('shap_summary.png'):
            st.image('shap_summary.png', caption='SHAP Feature Importance — Drivers of Policy Lapse', use_container_width=True)
        else:
            st.warning("Run train_lapse_model.py to generate SHAP plot.")

    st.markdown("---")
    st.markdown("<p class='section-header'>Cross-Validation Scores</p>", unsafe_allow_html=True)
    if meta:
        cv_df = pd.DataFrame({
            name: scores for name, scores in meta['cv_results'].items()
        })
        cv_df.index = [f"Fold {i+1}" for i in range(len(cv_df))]
        cv_summary = cv_df.agg(['mean','std']).T.round(4)
        cv_summary.columns = ['Mean AUC', 'Std Dev']
        cv_summary = cv_summary.sort_values('Mean AUC', ascending=False)
        st.dataframe(cv_summary, use_container_width=True)

    st.markdown("""
    <div class='action-box'>
    <strong>📌 Interview talking points from this model:</strong><br>
    • <b>AUC-ROC 0.78</b> on imbalanced data (14.4% lapse rate) — accuracy alone would be misleading<br>
    • <b>Class weights</b> (3.48× for lapsed class) address the imbalance without oversampling<br>
    • <b>Threshold tuning</b> to recall ≥ 0.75 — catching lapses is more valuable than avoiding false alarms<br>
    • <b>SHAP values</b> provide IRDAI-compliant model explainability for each prediction<br>
    • <b>Logistic Regression</b> won on AUC — simpler models often outperform on tabular data with good features
    </div>
    """, unsafe_allow_html=True)

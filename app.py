import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# =========================================================
# Load model
# =========================================================
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
EXPECTED_FEATURES = model.feature_names_in_

# =========================================================
# Custom CSS
# =========================================================
st.markdown(
    """
    <style>
    .risk-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        margin-top: 20px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# App Header
# =========================================================
st.title("💳 Credit Default Risk Prediction")
st.markdown(
    """
    This application estimates the **probability of credit default**
    using customer demographics and recent financial behaviour.
    """
)

# =========================================================
# Sidebar – Inputs
# =========================================================
with st.sidebar:
    st.header("Customer Information")

    LIMIT_BAL = st.number_input("Credit Limit (NTD)", min_value=0)
    AGE = st.number_input("Age (years)", min_value=18, max_value=100)

    EDUCATION_MAP = {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others"
    }

    EDUCATION = st.selectbox(
        "Education Level",
        options=list(EDUCATION_MAP.keys()),
        format_func=lambda x: EDUCATION_MAP[x]
    )

    EDUCATION_2 = 1 if EDUCATION == 2 else 0

    st.divider()

    REPAYMENT_LABELS = {
        -1: "Paid on time",
        0: "No delay",
        1: "1 month overdue",
        2: "2 months overdue",
        3: "3 months overdue",
        4: "4 months overdue",
        5: "5 months overdue",
        6: "6 months overdue",
        7: "7 months overdue",
        8: "8 months overdue",
        9: "9+ months overdue"
    }


    st.header("Repayment Status (Past 6 Months)")

    REPAY_OPTIONS = list(REPAYMENT_LABELS.keys())

    PAY_0 = st.selectbox(
        "September",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    PAY_2 = st.selectbox(
        "August",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    PAY_3 = st.selectbox(
        "July",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    PAY_4 = st.selectbox(
        "June",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    PAY_5 = st.selectbox(
        "May",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    PAY_6 = st.selectbox(
        "April",
        options=REPAY_OPTIONS,
        format_func=lambda x: REPAYMENT_LABELS[x]
    )

    st.divider()
    st.header("Billing & Payment Amounts (NTD)")

    BILL_AMT = []
    PAY_AMT = []

    for i in range(1, 7):
        BILL_AMT.append(
            st.number_input(f"Billing Amount Month {i}", min_value=0, key=f"bill{i}")
        )
        PAY_AMT.append(
            st.number_input(f"Payment Amount Month {i}", min_value=0, key=f"pay{i}")
        )

# =========================================================
# Feature Engineering (MATCHES TRAINING)
# =========================================================
pay_statuses = np.array([PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6])
BILL_AMT = np.array(BILL_AMT)
PAY_AMT = np.array(PAY_AMT)

PAY_MEAN = pay_statuses.mean()
PAY_TOTAL = np.sum(np.maximum(pay_statuses, 0))

BILL_MEAN = BILL_AMT.mean()
BILL_TOTAL = BILL_AMT.sum()
BILL_STD = BILL_AMT.std()

PAY_BILL_RATIO = PAY_AMT.sum() / (BILL_TOTAL + 1e-6)

# =========================================================
# Build model input (SAFE ORDERING)
# =========================================================
raw_input = pd.DataFrame([{
    "PAY_SEP": PAY_0,
    "PAY_AUG": PAY_2,
    "PAY_JUL": PAY_3,
    "PAY_JUN": PAY_4,
    "PAY_MAY": PAY_5,
    "PAY_APR": PAY_6,
    "PAY_MEAN": PAY_MEAN,
    "PAY_TOTAL": PAY_TOTAL,
    "BILL_STD": BILL_STD,
    "LIMIT_BAL": LIMIT_BAL,
    "PAY_BILL_RATIO": PAY_BILL_RATIO,
    "BILL_MEAN": BILL_MEAN,
    "BILL_TOTAL": BILL_TOTAL,
    "AGE": AGE,
    "EDUCATION_2": EDUCATION_2
}])

model_input = raw_input.reindex(columns=EXPECTED_FEATURES)

# =========================================================
# Prediction Section
# =========================================================
st.divider()
st.header("Prediction Result")

threshold = st.slider(
    "Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.05
)

if st.button("Predict Default Risk"):

    prob_default = model.predict_proba(model_input)[0, 1]
    risk_pct = prob_default * 100

    st.markdown(
        f"""
        <div class="risk-card">
            <h1>{risk_pct:.1f}%</h1>
            <p>Predicted Probability of Default</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(risk_pct / 100)

    if prob_default >= threshold:
        st.error("🔴 High Risk: Likely to Default")
    elif prob_default >= threshold * 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk: Unlikely to Default")

    st.caption(
        "The threshold is adjustable for demonstration purposes. "
        "Actual lending decisions depend on institutional risk appetite."
    )

    st.subheader("Model Input Summary")
    st.dataframe(
        model_input.T.rename(columns={0: "Value"}),
        use_container_width=True
    )

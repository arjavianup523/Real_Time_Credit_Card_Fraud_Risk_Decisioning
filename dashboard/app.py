import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.risk_decision import make_risk_decision


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_model.joblib"
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_model_metrics.csv"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "threshold_results.csv"
)


st.set_page_config(
    page_title="Fraud Risk Decisioning",
    page_icon="💳",
    layout="wide",
)


def find_raw_transaction_file():
    csv_files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DATA_DIR}"
        )

    preferred_names = [
        "transaction_data.csv",
        "transactions.csv",
        "fraud_transactions.csv",
    ]

    for preferred_name in preferred_names:
        preferred_path = (
            RAW_DATA_DIR / preferred_name
        )

        if preferred_path.exists():
            return preferred_path

    return csv_files[0]


@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_PATH
    )


@st.cache_data
def load_data():
    raw_data_path = (
        find_raw_transaction_file()
    )

    return pd.read_csv(
        raw_data_path
    )


@st.cache_data
def load_metrics():
    return pd.read_csv(
        METRICS_PATH
    )


@st.cache_data
def load_thresholds():
    return pd.read_csv(
        THRESHOLD_PATH
    )


def engineer_transaction_features(
    amount,
    merchant_category,
    country,
    device_type,
    is_new_device,
    is_new_location,
    transactions_last_1h,
    transactions_last_24h,
    avg_amount_last_30d,
    transaction_hour,
    customer_age,
    account_age_days,
    transaction_day_of_week,
    transaction_day,
    transaction_month,
):
    amount_deviation = (
        amount - avg_amount_last_30d
    ) / max(
        avg_amount_last_30d,
        1,
    )

    is_night = int(
        transaction_hour < 6
        or transaction_hour >= 22
    )

    amount_log = np.log1p(
        amount
    )

    amount_vs_customer_average = (
        amount
        / max(
            avg_amount_last_30d,
            1,
        )
    )

    velocity_ratio = (
        transactions_last_1h
        / max(
            transactions_last_24h,
            1,
        )
    )

    high_velocity_flag = int(
        transactions_last_1h >= 6
    )

    high_daily_activity_flag = int(
        transactions_last_24h > 20
    )

    young_customer_flag = int(
        customer_age < 25
    )

    new_account_flag = int(
        account_age_days < 90
    )

    device_location_risk = int(
        is_new_device == 1
        and is_new_location == 1
    )

    night_and_new_device = int(
        is_night == 1
        and is_new_device == 1
    )

    night_and_new_location = int(
        is_night == 1
        and is_new_location == 1
    )

    high_amount_and_new_device = int(
        amount > 50000
        and is_new_device == 1
    )

    return pd.DataFrame(
        [
            {
                "amount": amount,
                "merchant_category": merchant_category,
                "country": country,
                "device_type": device_type,
                "is_new_device": is_new_device,
                "is_new_location": is_new_location,
                "transactions_last_1h": transactions_last_1h,
                "transactions_last_24h": transactions_last_24h,
                "avg_amount_last_30d": avg_amount_last_30d,
                "amount_deviation": amount_deviation,
                "transaction_hour": transaction_hour,
                "is_night": is_night,
                "customer_age": customer_age,
                "account_age_days": account_age_days,
                "transaction_day_of_week": transaction_day_of_week,
                "transaction_day": transaction_day,
                "transaction_month": transaction_month,
                "amount_log": amount_log,
                "amount_vs_customer_average": amount_vs_customer_average,
                "velocity_ratio": velocity_ratio,
                "high_velocity_flag": high_velocity_flag,
                "high_daily_activity_flag": high_daily_activity_flag,
                "young_customer_flag": young_customer_flag,
                "new_account_flag": new_account_flag,
                "device_location_risk": device_location_risk,
                "night_and_new_device": night_and_new_device,
                "night_and_new_location": night_and_new_location,
                "high_amount_and_new_device": high_amount_and_new_device,
            }
        ]
    )


model = load_model()
df = load_data()
metrics = load_metrics()
thresholds = load_thresholds()


# ============================================================
# HEADER
# ============================================================

st.title(
    "💳 Real-Time Credit Card Fraud Detection & Risk Decisioning"
)

st.caption(
    "Production-style ML fraud detection with risk scoring and automated decisioning"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Transaction Simulator"
)

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=2500.0,
    step=500.0,
)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    sorted(
        df["merchant_category"].unique()
    ),
)

country = st.sidebar.selectbox(
    "Country",
    sorted(
        df["country"].unique()
    ),
)

device_type = st.sidebar.selectbox(
    "Device Type",
    sorted(
        df["device_type"].unique()
    ),
)

is_new_device = st.sidebar.selectbox(
    "New Device?",
    [0, 1],
    format_func=lambda x: (
        "Yes" if x == 1 else "No"
    ),
)

is_new_location = st.sidebar.selectbox(
    "New Location?",
    [0, 1],
    format_func=lambda x: (
        "Yes" if x == 1 else "No"
    ),
)

transactions_last_1h = st.sidebar.number_input(
    "Transactions in Last 1 Hour",
    min_value=0,
    value=2,
)

transactions_last_24h = st.sidebar.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=5,
)

avg_amount_last_30d = st.sidebar.number_input(
    "Average Amount — Last 30 Days",
    min_value=0.0,
    value=1500.0,
    step=100.0,
)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=14,
)

customer_age = st.sidebar.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=30,
)

account_age_days = st.sidebar.number_input(
    "Account Age (Days)",
    min_value=0,
    value=500,
)

transaction_day_of_week = st.sidebar.slider(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=2,
)

transaction_day = st.sidebar.slider(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15,
)

transaction_month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6,
)


# ============================================================
# PREDICTION
# ============================================================

transaction_features = (
    engineer_transaction_features(
        amount=amount,
        merchant_category=merchant_category,
        country=country,
        device_type=device_type,
        is_new_device=is_new_device,
        is_new_location=is_new_location,
        transactions_last_1h=transactions_last_1h,
        transactions_last_24h=transactions_last_24h,
        avg_amount_last_30d=avg_amount_last_30d,
        transaction_hour=transaction_hour,
        customer_age=customer_age,
        account_age_days=account_age_days,
        transaction_day_of_week=transaction_day_of_week,
        transaction_day=transaction_day,
        transaction_month=transaction_month,
    )
)

fraud_probability = float(
    model.predict_proba(
        transaction_features
    )[0][1]
)

decision = make_risk_decision(
    fraud_probability
)


# ============================================================
# LIVE DECISION
# ============================================================

st.header(
    "Live Transaction Decision"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%",
    )

with col2:
    st.metric(
        "Risk Score",
        f"{decision['risk_score']}/100",
    )

with col3:
    st.metric(
        "Risk Level",
        decision["risk_level"],
    )

with col4:
    st.metric(
        "Decision",
        decision["decision"],
    )


if decision["decision"] == "APPROVE":

    st.success(
        "🟢 LOW RISK — TRANSACTION APPROVED"
    )

elif decision["decision"] == "REVIEW":

    st.warning(
        "🟡 MEDIUM RISK — TRANSACTION SENT FOR REVIEW / OTP"
    )

else:

    st.error(
        "🔴 HIGH RISK — TRANSACTION BLOCKED"
    )


st.write(
    "Review threshold: "
    f"{decision['review_threshold'] * 100:.2f}%"
)

st.write(
    "Block threshold: "
    f"{decision['block_threshold'] * 100:.2f}%"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header(
    "Model Performance"
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "Precision",
        f"{metrics.iloc[0]['precision']:.3f}",
    )

with m2:
    st.metric(
        "Recall",
        f"{metrics.iloc[0]['recall']:.3f}",
    )

with m3:
    st.metric(
        "F1 Score",
        f"{metrics.iloc[0]['f1_score']:.3f}",
    )

with m4:
    st.metric(
        "ROC-AUC",
        f"{metrics.iloc[0]['roc_auc']:.3f}",
    )

with m5:
    st.metric(
        "PR-AUC",
        f"{metrics.iloc[0]['pr_auc']:.3f}",
    )


# ============================================================
# FRAUD ANALYTICS
# ============================================================

st.header(
    "Fraud Analytics"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Transactions",
        f"{len(df):,}",
    )

with col2:

    fraud_count = int(
        df["fraud"].sum()
    )

    st.metric(
        "Fraud Transactions",
        f"{fraud_count:,}",
    )

with col3:

    fraud_rate = (
        df["fraud"].mean() * 100
    )

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%",
    )


# ============================================================
# FRAUD BY HOUR
# ============================================================

hour_analysis = (
    df.groupby(
        "transaction_hour",
        as_index=False,
    )["fraud"]
    .mean()
    .rename(
        columns={
            "fraud": "fraud_rate"
        }
    )
)

hour_analysis["fraud_rate"] *= 100

fig_hour = px.line(
    hour_analysis,
    x="transaction_hour",
    y="fraud_rate",
    markers=True,
    title="Fraud Rate by Transaction Hour",
)

fig_hour.update_layout(
    xaxis_title="Hour",
    yaxis_title="Fraud Rate (%)",
)

st.plotly_chart(
    fig_hour,
    use_container_width=True,
)


# ============================================================
# FRAUD BY DEVICE
# ============================================================

device_analysis = (
    df.groupby(
        "device_type",
        as_index=False,
    )["fraud"]
    .mean()
    .rename(
        columns={
            "fraud": "fraud_rate"
        }
    )
)

device_analysis["fraud_rate"] *= 100

fig_device = px.bar(
    device_analysis,
    x="device_type",
    y="fraud_rate",
    title="Fraud Rate by Device",
)

fig_device.update_layout(
    xaxis_title="Device",
    yaxis_title="Fraud Rate (%)",
)

st.plotly_chart(
    fig_device,
    use_container_width=True,
)


# ============================================================
# FRAUD BY COUNTRY
# ============================================================

country_analysis = (
    df.groupby(
        "country",
        as_index=False,
    )["fraud"]
    .mean()
    .rename(
        columns={
            "fraud": "fraud_rate"
        }
    )
    .sort_values(
        "fraud_rate",
        ascending=False,
    )
)

country_analysis["fraud_rate"] *= 100

fig_country = px.bar(
    country_analysis,
    x="country",
    y="fraud_rate",
    title="Fraud Rate by Country",
)

fig_country.update_layout(
    xaxis_title="Country",
    yaxis_title="Fraud Rate (%)",
)

st.plotly_chart(
    fig_country,
    use_container_width=True,
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

st.header(
    "Fraud Threshold Optimization"
)

threshold_chart = px.line(
    thresholds,
    x="threshold",
    y=[
        "precision",
        "recall",
        "f1_score",
    ],
    markers=True,
    title="Precision / Recall / F1 by Threshold",
)

threshold_chart.update_layout(
    xaxis_title="Decision Threshold",
    yaxis_title="Score",
)

st.plotly_chart(
    threshold_chart,
    use_container_width=True,
)


# ============================================================
# BUSINESS DECISION FRAMEWORK
# ============================================================

st.header(
    "Business Decision Framework"
)

st.markdown(
    """
### APPROVE

Low predicted fraud probability.

### REVIEW

Medium-risk transaction requiring additional verification,
such as OTP or manual review.

### BLOCK

High predicted fraud probability and automatically rejected.

The ML model produces the fraud probability.
The business decision engine converts that probability
into an operational action.
"""
)
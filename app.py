import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.caption("Trained with MLflow | Deployed with Streamlit")

# ----------------------------
# Contract encoding
# MUST match training exactly
# ----------------------------
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "model", "model.pkl")
    return joblib.load(model_path)

model = load_model()

# ----------------------------
# User inputs
# ----------------------------
st.subheader("Customer Details")

tenure = st.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract_label = st.selectbox(
    "Contract Type",
    list(contract_map.keys())
)

# Convert contract label to numeric value
contract = contract_map[contract_label]

# ----------------------------
# Build input DataFrame
# MUST match training features
# ----------------------------
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract
}])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🚨 Churn: YES (Customer likely to leave)")
    else:
        st.success("✅ Churn: NO (Customer likely to stay)")

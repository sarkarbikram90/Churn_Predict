import streamlit as st
import pandas as pd
import joblib
import os

# App configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.caption("Predict whether a customer will churn based on their details.")
st.caption("Provide the customer information and click 'Predict' to see the result.")
st.caption("Developed by Bikram with Python, Scikit-learn, MLflow, and Streamlit.")
st.caption("The model is trained locally using MLflow for experiment tracking and then deployed as a lightweight Streamlit web app for predictions.")

# Contract encoding
# MUST match training exactly
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

# Load trained model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "model", "model.pkl")
    return joblib.load(model_path)

model = load_model()

# User inputs
st.subheader("Customer Details")

tenure = st.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=0
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=0.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=0.0
)

contract_label = st.selectbox(
    "Contract Type",
    list(contract_map.keys())
)

# Convert contract label to numeric value
contract = contract_map[contract_label]

# Build input DataFrame

# Feature engineering (must match training)
avg_monthly_spend = total / (tenure + 1)
is_long_tenure = 1 if tenure > 24 else 0

# MUST match training features
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "AvgMonthlySpend": avg_monthly_spend,
    "IsLongTenure": is_long_tenure
}])


# Prediction
## if st.button("Predict"):
    ## prediction = model.predict(input_df)[0]

    ## if prediction == 1:
    ##     st.error("🚨 Churn: YES (Customer likely to leave)")
    ## else:
    ##     st.success("✅ Churn: NO (Customer likely to stay)")

probability = model.predict_proba(input_df)[0][1]
st.info(f"Customer Churn Probability Score(%): {probability:.2%}")

if probability > 0.7:
    st.error("High churn risk 🚨")
elif probability > 0.4:
    st.warning("Medium churn risk ⚠️")
else:
    st.success("Low churn risk ✅")


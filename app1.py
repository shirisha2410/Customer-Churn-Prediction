import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# Title
st.title("Customer Churn Prediction")
st.write("Enter customer details")


# Input fields
gender = st.selectbox(
    "Gender",
    label_encoders["Gender"].classes_
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure_months = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)

contract = st.selectbox(
    "Contract",
    label_encoders["Contract"].classes_
)

internet_service = st.selectbox(
    "Internet Service",
    label_encoders["InternetService"].classes_
)

payment_method = st.selectbox(
    "Payment Method",
    label_encoders["PaymentMethod"].classes_
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=75.5
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    max_value=20,
    value=3
)


# Prediction button
if st.button("Predict Churn"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "SeniorCitizen": senior_citizen,
        "TenureMonths": tenure_months,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "SupportCalls": support_calls
    }])

    # Encode categorical columns
    categorical_columns = [
        "Gender",
        "Contract",
        "InternetService",
        "PaymentMethod"
    ]

    for col in categorical_columns:
        input_data[col] = label_encoders[col].transform(
            input_data[col]
        )

    # Arrange columns in correct order
    input_data = input_data[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to NOT CHURN")

    st.write(f"Churn Probability: {probability:.2%}")
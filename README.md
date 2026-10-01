
# Customer Churn Prediction

## Live Demo

[Try the Customer Churn Prediction App]

(https://customer-churn-prediction-ghwumdrdnnfvnemdldk4ak.streamlit.app)

## Project Overview

This project predicts whether a customer is likely to churn using Machine Learning.

The project includes data preprocessing, categorical encoding, feature scaling, model training, evaluation, and deployment using Streamlit.

## Business Problem

Customer churn is an important business problem. Identifying customers who are likely to leave can help companies take appropriate customer retention actions.

## Features

The model uses the following customer information:

- Gender
- Age
- Senior Citizen
- Tenure Months
- Contract
- Internet Service
- Payment Method
- Monthly Charges
- Support Calls

## Data Preprocessing

The following preprocessing techniques were used:

- Categorical feature encoding using LabelEncoder
- Numerical feature scaling using StandardScaler
- Feature order validation before prediction

## Machine Learning Model

The project uses:

- Logistic Regression
- StandardScaler
- LabelEncoder
- Scikit-learn

## Model Prediction

The application provides:

- Churn prediction
- Churn probability

Example:

Customer is likely to CHURN

Churn Probability: 64.03%

## Streamlit Application

An interactive Streamlit web application was developed where users can enter customer details and get a real-time churn prediction.

## Deployment

The application is deployed using Streamlit Community Cloud.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── README.md
├── app1.py
├── churn_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_columns.pkl
└── requirements.txt

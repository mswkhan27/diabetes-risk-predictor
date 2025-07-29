
import streamlit as st
import pandas as pd
import numpy as np
from risk_predictor import RiskScorePredictor
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Risk Score Predictor")
st.write("This app uses a fine-tuned Random Forest model trained on the Pima Indians Diabetes dataset to predict diabetes risk with ML model with 71% accuracy..")

# Load model
predictor = RiskScorePredictor()
X_train, X_test, y_train, y_test = predictor.load_and_prepare_data("diabetes.csv", "Outcome")
predictor.train(X_train, y_train)

st.subheader("📝 Input Patient Data for Prediction")

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=85.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=28.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=10, max_value=120, value=33)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])
        input_scaled = predictor.scaler.transform(input_data)
        prediction = predictor.model.predict(input_scaled)[0]
        prediction_proba = predictor.model.predict_proba(input_scaled)[0][1]

        st.success(f"Prediction: {'Diabetic (High Risk)' if prediction == 1 else 'Non-Diabetic (Low Risk)'}")
        st.info(f"Predicted Probability of Diabetes: {prediction_proba:.2f}")

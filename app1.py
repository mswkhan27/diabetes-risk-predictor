import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("diabetes_model.keras")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🩺 Diabetes Risk Predictor")
st.markdown("This app predicts the risk of diabetes based on health data using a neural network with 83% accuracy.")

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
    glucose = st.number_input("Glucose", 50, 200, step=1)
    blood_pressure = st.number_input("Blood Pressure", 30, 140, step=1)
    skin_thickness = st.number_input("Skin Thickness", 7, 100, step=1)
    insulin = st.number_input("Insulin", 15, 846, step=1)
    bmi = st.number_input("BMI", 10.0, 70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
    age = st.number_input("Age", 10, 100, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    risk = "🔴 High Risk" if prediction > 0.5 else "🟢 Low Risk"
    
    st.subheader("Result")
    st.write(f"**Prediction:** {risk}")
    st.write(f"**Confidence:** {prediction:.2%}")
    st.progress(int(prediction * 100))

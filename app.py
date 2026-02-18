import streamlit as st
import numpy as np
import joblib

# Load trained model (or pipeline if saved as pipeline)
model = joblib.load("model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

# ---------------------------
# User Inputs
# ---------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=50)

cp = st.selectbox(
    "Chest Pain Type (CP)",
    options=[1, 2, 3, 4],
    help="1 = Typical Angina, 4 = Asymptomatic"
)

chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    help="0 = No, 1 = Yes"
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[1, 2, 3],
    help="1 = Upsloping, 2 = Flat, 3 = Downsloping"
)

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict"):
    features = np.array([[age, cp, chol, thalach, exang, oldpeak, slope]])
    
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write("Model Used: AdaBoost with Decision Tree (max_depth=3)")

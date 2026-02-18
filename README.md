# ❤️ Heart Disease Prediction App

A Machine Learning web application that predicts the likelihood of heart disease using an AdaBoost ensemble model.

Built with:
- Python
- Scikit-learn
- Streamlit
- Joblib

---

## 📌 Project Overview

This project uses a supervised machine learning approach to predict whether a patient is at high or low risk of heart disease based on clinical features.

The model was trained using an AdaBoost classifier with a Decision Tree base estimator (max_depth=3), optimized for improved predictive performance.

The trained model was serialized into a `.pkl` file using Sathvi and deployed using Streamlit.

---

## 🧠 Model Details

Algorithm: AdaBoostClassifier  
Base Learner: DecisionTreeClassifier (max_depth=3)  
Number of Estimators: 200  
Learning Rate: 0.8  

Features used:

- Age
- CP (Chest Pain Type)
- Chol (Cholesterol)
- Thalach (Maximum Heart Rate Achieved)
- Exang (Exercise Induced Angina)
- Oldpeak (ST Depression)
- Slope (Slope of Peak Exercise ST Segment)

---

## 🚀 Live App

You can run the app locally using:

```bash
pip install -r requirements.txt
streamlit run app.py

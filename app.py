import streamlit as st
import pickle
import numpy as np


with open("xgbModel.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Cross Insurance Prediction")


st.sidebar.header("Enter Details")

gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
vehicle_age = st.sidebar.selectbox("Vehicle Age", options=[0, 1, 2], format_func=lambda x: "<1 Year" if x == 0 else ("1-2 Years" if x == 1 else ">2 Years"))
vehicle_damage = st.sidebar.selectbox("Vehicle Damage", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Numerical Inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
driving_license = st.sidebar.radio("Driving License", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
region_code = st.sidebar.number_input("Region Code", min_value=0, max_value=52, step=1)
previously_insured = st.sidebar.radio("Previously Insured", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
annual_premium = st.sidebar.number_input("Annual Premium", min_value=1000, max_value=100000, step=100)
vintage = st.sidebar.number_input("Vintage (Days with Policy)", min_value=0, max_value=500, step=1)


threshold = st.sidebar.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.74, step=0.01)


features = np.array([[gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, vintage]])


if st.sidebar.button("Predict"):
    probabilities = model.predict_proba(features)[:, 1]  # Get probability of positive class
    prediction = (probabilities >= threshold).astype(int)  # Apply threshold

    st.write(f"**Predicted Insurance CrossOver Status:** {'Approved' if prediction[0] == 1 else 'Rejected'}")
    st.write(f"**Prediction Probability:** {probabilities[0]:.4f}")
    st.write(f"**Threshold Used:** {threshold}")

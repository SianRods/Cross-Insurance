import streamlit as st
import pickle
import numpy as np


with open("models/xgbModel.pkl", "rb") as file:
    model = pickle.load(file)


background_image1 = "https://images.unsplash.com/39/lIZrwvbeRuuzqOoWJUEn_Photoaday_CSD%20%281%20of%201%29-5.jpg?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 

background_image2="https://images.pexels.com/photos/1420019/pexels-photo-1420019.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
st.markdown(
    f"""
    <style>
    .stApp{{

        background: url('{background_image2}') no-repeat center center fixed;
        background-size: cover;
        color:red;
    }}  
    .top-bar {{
        background-color: #1f77b4;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: white;
        border-radius: 8px;
        margin-bottom: 10px;
    }}
    .top-bar:hover{{
        color:black;
    }}
    </style>
    <div class="top-bar">Welcome !</div>
    """,
    unsafe_allow_html=True
)

st.title("Cross Insurance Prediction")

st.header("Enter Details")


def red_label(text):
    return f"<span style='color:red; font-weight:bold;'>{text}</span>"

st.markdown("<h3 style='color:red;'>Categorical Inputs</h3>", unsafe_allow_html=True) 



gender = st.selectbox(red_label("Gender"), options=["Male", "Female"])
vehicle_age = st.selectbox(red_label("Vehicle Age"), options=["<1 Year", "1-2 Years", ">2 Years"])
vehicle_damage = st.selectbox(red_label("Vehicle Damage"), options=["Yes", "No"])


age = st.number_input(red_label("Age"), min_value=18, max_value=100, step=1)
driving_license = st.radio(red_label("Driving License"), [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
region_code = st.number_input(red_label("Region Code"), min_value=0, max_value=52, step=1)
previously_insured = st.radio(red_label("Previously Insured"), [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
annual_premium = st.number_input(red_label("Annual Premium"), min_value=1000, max_value=100000, step=100)
vintage = st.number_input(red_label("Vintage (Days with Policy)"), min_value=0, max_value=500, step=1)


threshold = st.slider(red_label("Prediction Threshold"), min_value=0.0, max_value=1.0, value=0.74, step=0.01)


gender = 1 if gender == "Male" else 0
vehicle_age = 0 if vehicle_age == "<1 Year" else (1 if vehicle_age == "1-2 Years" else 2)
vehicle_damage = 1 if vehicle_damage == "Yes" else 0
driving_license = 1 if driving_license == "Yes" else 0
previously_insured = 1 if previously_insured == "Yes" else 0


features = np.array([[gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, vintage]])

if st.button("Predict"):
    probabilities = model.predict_proba(features)[:, 1]  
    prediction = (probabilities >= threshold).astype(int)  

    st.markdown(f"""
    <p style="color: white; font-size: 18px;">
        <b>Predicted Insurance CrossOver Status:</b> {'Approved' if prediction[0] == 1 else 'Rejected'}
    </p>
    <p style="color: white; font-size: 18px;">
        <b>Prediction Probability:</b> {probabilities[0]:.4f}
    </p>
    <p style="color: white; font-size: 18px;">
        <b>Threshold Used:</b> {threshold}
    </p>
""", unsafe_allow_html=True)


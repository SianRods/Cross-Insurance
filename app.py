import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("models/xgbModel.pkl", "rb") as file:
    model = pickle.load(file)

# Custom CSS for background image
background_image = "https://images.unsplash.com/39/lIZrwvbeRuuzqOoWJUEn_Photoaday_CSD%20%281%20of%201%29-5.jpg?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with the path to your image
st.markdown(
    f"""
    <style>
    .stApp,body {{
        color:red !important,
        background: url('{background_image}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Cross Insurance Prediction")

st.header("Enter Details")

# Categorical Inputs
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
vehicle_age = st.selectbox("Vehicle Age", options=[0, 1, 2], format_func=lambda x: "<1 Year" if x == 0 else ("1-2 Years" if x == 1 else ">2 Years"))
vehicle_damage = st.selectbox("Vehicle Damage", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Numerical Inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)
driving_license = st.radio("Driving License", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
region_code = st.number_input("Region Code", min_value=0, max_value=52, step=1)
previously_insured = st.radio("Previously Insured", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
annual_premium = st.number_input("Annual Premium", min_value=1000, max_value=100000, step=100)
vintage = st.number_input("Vintage (Days with Policy)", min_value=0, max_value=500, step=1)

# Prediction Threshold
threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.74, step=0.01)

# Prepare input features
features = np.array([[gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, vintage]])

if st.button("Predict"):
    probabilities = model.predict_proba(features)[:, 1]  # Get probability of positive class
    prediction = (probabilities >= threshold).astype(int)  # Apply threshold

    st.write(f"**Predicted Insurance CrossOver Status:** {'Approved' if prediction[0] == 1 else 'Rejected'}")
    st.write(f"**Prediction Probability:** {probabilities[0]:.4f}")
    st.write(f"**Threshold Used:** {threshold}")

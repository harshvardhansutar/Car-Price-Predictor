import streamlit as st
import pandas as pd
import joblib

# Load trained model (Random Forest pipeline)
model = joblib.load("RandomForest_pipeline.pkl")

st.title("🚗 Car Price Prediction App")
st.write("Enter the details below to get an estimated selling price for your car.")

# Input fields
present_price = st.number_input("Present/Showroom Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
years_old = st.number_input("Car Age (Years)", min_value=0, step=1)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):
    # Arrange input into DataFrame
    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Years_Old': [years_old],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission]
    })

    # Predict
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:.2f} lakhs")

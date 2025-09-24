import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model (Random Forest pipeline)
model = joblib.load("RandomForest_pipeline.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction App")
st.write("Enter details below OR upload a CSV file to get estimated car prices.")

# Sidebar Inputs
st.sidebar.header("Car Details Input")

present_price = st.sidebar.number_input("Present/Showroom Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])
years_old = st.sidebar.number_input("Car Age (Years)", min_value=0, step=1)

fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# Prediction Button
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Years_Old': [years_old],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission]
    })

    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:.2f} lakhs")

# ============================
# 2️⃣ Bulk Prediction via CSV Upload
# ============================
st.subheader("📂 Bulk Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with car details", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:", df.head())

    try:
        predictions = model.predict(df)
        df["Predicted_Price (Lakhs)"] = predictions
        st.success("✅ Predictions completed!")
        st.write(df)

        # Download option
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv_out, file_name="predicted_prices.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================
# 3️⃣ Feature Importance Chart
# ============================
st.subheader("📊 Feature Importance (Random Forest)")

try:
    # Extract preprocessor & model
    preprocessor = model.named_steps['preprocessor']
    rf_model = model.named_steps['model']

    # Get feature names
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(['Fuel_Type', 'Seller_Type', 'Transmission'])
    feature_names = ['Present_Price', 'Kms_Driven', 'Owner', 'Years_Old'] + list(cat_features)

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not load feature importance: {e}")

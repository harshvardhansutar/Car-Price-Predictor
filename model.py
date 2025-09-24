import joblib
import numpy as np

# Load the best model (Random Forest here, you can change if needed)
model = joblib.load("RandomForest_pipeline.pkl")

# Define feature order
numeric_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Years_Old']
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']

def predict_price():
    print("\nEnter Car Details to Predict Selling Price:")

    present_price = float(input("Enter Present/Showroom Price (in lakhs): "))
    kms_driven = int(input("Enter Kilometers Driven: "))
    owner = int(input("Number of Previous Owners (0, 1, 2, etc.): "))
    years_old = int(input("Car Age in Years: "))

    fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ")
    seller_type = input("Seller Type (Dealer/Individual): ")
    transmission = input("Transmission (Manual/Automatic): ")

    # Arrange input in dataframe format
    input_data = {
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Years_Old': [years_old],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission]
    }

    import pandas as pd
    input_df = pd.DataFrame(input_data)

    # Predict using trained pipeline
    predicted_price = model.predict(input_df)[0]
    print(f"\nEstimated Selling Price: ₹ {predicted_price:.2f} lakhs")

# Run prediction function
if __name__ == "__main__":
    predict_price()

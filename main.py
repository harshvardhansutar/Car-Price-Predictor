import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("P:/ShadowFox AIML Internship/Level 2/car.csv")

print("Summary : \n", data.describe())
print("\n\nFirst few Rows : \n", data.head())
print("Checking Null Values : \n", data.isnull().sum())
print("Columns : \n\n", data.columns)

# Feature Engineering
data['Years_Old'] = 2025 - data["Year"]
data.drop(["Year", "Car_Name"], axis=1, inplace=True)

# Target and Features
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Train / Test Split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

# Preprocessing
numeric_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Years_Old']
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']

numerical_transform = StandardScaler()
categorical_transform = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transform, numeric_features),
        ('cat', categorical_transform, categorical_features)
    ]
)

# Models 
models = {
    "LinearRegression" : LinearRegression(),
    "RandomForest" : RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting" : GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = []

for name,model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(x_train,y_train)
    preds = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

    # save trained pipeline
    joblib.dump(pipeline, f"{name}_pipeline.pkl")

# Results 
results_data = pd.DataFrame(results, columns=["Model", 'MAE', 'RMSE', 'R2'])
print("\n\nModel Evaluation : ")
print(results_data)


# Feature Importance (using RandomForest)
best_model = joblib.load("RandomForest_pipeline.pkl")

# Extract feature names after preprocessing
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat']
feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))

importances = best_model.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({"Feature":feature_names, "Importance" : importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance (Random Forest)")
plt.show()

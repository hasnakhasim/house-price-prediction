from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import joblib
import os

# Load house dataset
data = pd.read_csv("houses.csv")

# Features and target
X = data[[
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age"
]]

y = data["price_lakh"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# Different ML algorithm
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2 Score:", r2_score(y_test, predictions))

# Save as a separate model file
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_model.pkl")

print("House Price model trained successfully!")
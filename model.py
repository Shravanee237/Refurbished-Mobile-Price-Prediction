import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("used_device_data.csv")

# Remove missing values
df = df.dropna()

# 🎯 TARGET (correct column)
y = df["normalized_used_price"]

# FEATURES
X = df.drop(["normalized_used_price"], axis=1)

# One-hot encoding
X = pd.get_dummies(X)

# Save columns
pickle.dump(X.columns, open("model_columns.pkl", "wb"))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully")
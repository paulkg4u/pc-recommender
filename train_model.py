import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


# Load data
df = pd.read_csv("builds.csv")

print(df.head())

# Encode categorical feature 'optimal_for'
label_encoder = LabelEncoder()
df["optimal_for_encoded"] = label_encoder.fit_transform(df["optimal_for"])

# Features (price & encoded category) and targets (CPU, GPU, RAM)
X = df[["price", "optimal_for_encoded"]]
y = df[["cpu_id", "gpu_id", "ram_id"]]  # Multi-output target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-output regression model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, "build_recommender.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and encoder saved successfully!")


# Function to recommend build

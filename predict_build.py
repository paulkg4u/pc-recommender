import pandas as pd
import joblib
import numpy as np

# Load model & label encoder
model = joblib.load("build_recommender.pkl")
label_encoder = joblib.load("label_encoder.pkl")
components_df = pd.read_csv("components.csv")

# Function to get component details
def get_component_details(component_id):
    component = components_df[components_df["component_id"] == component_id]
    if not component.empty:
        return {
            "id": int(component["component_id"].values[0]),
            "name": component["component_name"].values[0],
            "type": component["component_type"].values[0],
            "price": int(component["price"].values[0]),
            "performance_score": int(component["performance_score"].values[0]),
        }
    return None

# Function to recommend a build
def recommend_build(price, optimal_for):
    optimal_for_encoded = label_encoder.transform([optimal_for])[0]
    X_input = pd.DataFrame([[price, optimal_for_encoded]], columns=["price", "optimal_for_encoded"])

    # Predict component IDs
    predicted_components = model.predict(X_input)
    print(predicted_components)
    cpu_id, gpu_id, ram_id = predicted_components[0].astype(int)

    return {
        "CPU": get_component_details(cpu_id),
        "GPU": get_component_details(gpu_id),
        "RAM": get_component_details(ram_id),
    }


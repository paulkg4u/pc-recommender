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
    X_input = pd.DataFrame([[price, optimal_for_encoded]], columns=[
                           "price", "optimal_for_encoded"])

    # Predict component IDs
    predicted_components_scores = model.predict(X_input)

    cpu_performance_score, gpu_performance_score, ram_performance_score = predicted_components_scores[0].astype(
        int)

    # from components.csv, get component ids matching the required performance scores, so that the total price does not go over the price

    cpu_id = components_df[(components_df["component_type"] == "CPU") &
                           (components_df["performance_score"] >= cpu_performance_score) &
                           (components_df["price"] <= price)].sort_values("price", ascending=True)["component_id"].values[0]
    remaining_price = price - \
        components_df[components_df["component_id"]
                      == cpu_id]["price"].values[0]
    gpu_id = components_df[(components_df["component_type"] == "GPU") &
                           (components_df["performance_score"] >= gpu_performance_score) &
                           (components_df["price"] <= remaining_price)].sort_values("price", ascending=True)["component_id"].values[0]
    remaining_price = remaining_price - \
        components_df[components_df["component_id"]
                      == gpu_id]["price"].values[0]
    ram_id = components_df[(components_df["component_type"] == "RAM") &
                           (components_df["performance_score"] >= ram_performance_score) &
                           (components_df["price"] <= remaining_price)].sort_values("price", ascending=True)["component_id"].values[0]

    if not cpu_id or not gpu_id or not ram_id:
        return {
            'success': False,
            'error': 'Could not find components matching the requirement and budget'
        }



    return {
        'success': True,
        'data': {
            "CPU": get_component_details(cpu_id),
            "GPU": get_component_details(gpu_id),
            "RAM": get_component_details(ram_id),
        }
    }

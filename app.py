from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from predict_build import recommend_build
app = Flask(__name__)


label_encoder = joblib.load("label_encoder.pkl")
model = joblib.load("build_recommender.pkl")
@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        # Get query parameters
        user_budget = float(request.args.get("budget"))
        user_optimal_for = request.args.get("optimal_for")

        # Encode the optimal_for category
        

        # Return JSON response
        response = recommend_build(user_budget, user_optimal_for)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
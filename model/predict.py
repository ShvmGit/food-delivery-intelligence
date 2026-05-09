"""
Model Prediction Module
Loads the trained model and provides prediction functionality.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'delivery_time_model.pkl'
META_PATH = Path(__file__).parent.parent / 'model_metadata.json'


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run 'py model/train.py' first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata():
    """Load model metadata (features, metrics, type)."""
    if not META_PATH.exists():
        return None
    with open(META_PATH, 'r') as f:
        return json.load(f)


def predict_delivery_time(model, input_data: dict) -> dict:
    """
    Predict delivery time from input parameters.

    Args:
        model: Trained sklearn model
        input_data: Dict with feature values

    Returns:
        Dict with prediction and delay category
    """
    metadata = load_metadata()
    features = metadata['features'] if metadata else [
        'distance_km', 'pickup_delay_min', 'traffic_score', 'weather_score',
        'delivery_person_age', 'delivery_person_ratings', 'vehicle_condition',
        'multiple_deliveries', 'order_hour', 'is_peak_hour',
    ]

    # Build DataFrame with correct feature order
    df = pd.DataFrame([{f: input_data.get(f, 0) for f in features}])
    prediction = model.predict(df)[0]

    # Classify delay
    if prediction < 20:
        category = "🟢 Fast Delivery"
        color = "green"
    elif prediction < 35:
        category = "🟡 Normal Delivery"
        color = "orange"
    else:
        category = "🔴 Delayed Delivery"
        color = "red"

    # Generate operational insights
    insights = []
    if input_data.get('traffic_score', 0) >= 3:
        insights.append("High traffic conditions are likely increasing delivery time")
    if input_data.get('weather_score', 0) >= 3:
        insights.append("Adverse weather may be contributing to delays")
    if input_data.get('multiple_deliveries', 0) >= 2:
        insights.append("Multiple deliveries in this route may cause inefficiencies")
    if input_data.get('delivery_person_ratings', 0) >= 4.5:
        insights.append("High-rated delivery person should help maintain efficiency")
    if input_data.get('vehicle_condition', 1) == 0:
        insights.append("Poor vehicle condition may be affecting performance")

    if not insights:
        insights.append("Conditions look favorable for timely delivery")

    return {
        'prediction': round(prediction, 1),
        'category': category,
        'color': color,
        'insights': insights,
    }

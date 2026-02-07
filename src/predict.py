import os
import pandas as pd
import joblib
from src.preprocess import preprocess_data

def predict(input_data: dict):
    # Fix model path for deployment
    BASE_DIR = os.path.dirname(__file__)           # src folder
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")  # go up one level

    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])
    X_scaled, _ = preprocess_data(df, training=False)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return prediction, probability

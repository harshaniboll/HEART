from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Machine Learning API for Heart Disease Prediction",
    version="1.0"
)

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

HTML_PATH = os.path.join(
    BASE_DIR,
    "public",
    "index.html"
)

# ==============================
# LOAD MODEL
# ==============================

with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]


# ==============================
# INPUT DATA
# ==============================

class PatientData(BaseModel):
    data: dict


# ==============================
# DASHBOARD
# ==============================

@app.get("/")
def home():
    return FileResponse(HTML_PATH)


# ==============================
# API TEST
# ==============================

@app.get("/api")
def api_home():
    return {
        "message": "Heart Disease Prediction API is running successfully"
    }


# ==============================
# PREDICTION
# ==============================

@app.post("/predict")
def predict(patient: PatientData):

    input_data = pd.DataFrame([patient.data])

    # Keep the same feature order used during training
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get confidence
    probability = model.predict_proba(input_scaled)[0].max()

    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(float(probability) * 100, 2)
    }
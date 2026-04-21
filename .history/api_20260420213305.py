from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging

import joblib
import os

# create folder if not exists
os.makedirs("model", exist_ok=True)

# save model
joblib.dump(model, "model/model.pkl")

# save columns
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("✅ Model saved successfully")

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Initialize FastAPI App
# -------------------------------
app = FastAPI(title="Loan Prediction API", version="1.0")

# -------------------------------
# File Paths
# -------------------------------
MODEL_PATH = "model/model.pkl"
COLUMNS_PATH = "model/columns.pkl"

# -------------------------------
# Load Model & Columns Safely
# -------------------------------
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found!")

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError("model.pkl is empty!")

    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError("columns.pkl not found!")

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)

    logger.info("Model and columns loaded successfully")
    return model, columns


model, columns = load_artifacts()

# -------------------------------
# Input Schema (VERY IMPORTANT)
# -------------------------------
class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


# -------------------------------
# Home Route
# -------------------------------
@app.get("/")
def home():
    return {"message": "🚀 Loan Prediction API is running successfully"}


# -------------------------------
# Prediction Route
# -------------------------------
@app.post("/predict")
def predict(data: LoanInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # One-hot encoding
        df = pd.get_dummies(df)

        # Align columns with training data
        df = df.reindex(columns=columns, fill_value=0)

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        logger.info(f"Prediction made: {prediction}, Probability: {probability}")

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Health Check Route (Optional)
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "OK"}
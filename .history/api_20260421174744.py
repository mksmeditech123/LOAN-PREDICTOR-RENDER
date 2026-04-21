from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI()

print("🚀 Loading model...")

# ==============================
# SAFE PATH (VERY IMPORTANT)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

# Load model
model = joblib.load(model_path)
columns = joblib.load(columns_path)

print("✅ Model loaded successfully")

# ==============================
# ROUTES
# ==============================
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # Encoding
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}
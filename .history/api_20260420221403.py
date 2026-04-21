from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

print("🚀 Loading model...")

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

print("✅ Model loaded")

@app.get("/")
def home():
    return {"message": "API is running"}

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
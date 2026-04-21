float(prob)
        }

    except Exception as e:
        return {"error": str(e)}import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Use your dataset path here
df = pd.read_csv("loan_data.csv")

# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
df = df.dropna()

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# -------------------------------
# 3. TARGET VARIABLE
# -------------------------------
# Convert Loan_Status (Y/N → 1/0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# -------------------------------
# 4. FEATURES & TARGET
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -------------------------------
# 5. ENCODING (IMPORTANT)
# -------------------------------
X = pd.get_dummies(X)

# -------------------------------
# 6. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 8. SAVE MODEL
# -------------------------------
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# -------------------------------
# 9. SAVE COLUMNS
# -------------------------------
with open("model/columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and columns saved successfully!")
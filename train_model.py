import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting training...")

# Load dataset
df = pd.read_csv("loan_data.csv")

# Drop missing values
df = df.dropna()

# Drop ID column if exists
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# Convert target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Encode categorical variables
X = pd.get_dummies(X)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns
with open("model/columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model saved successfully!")
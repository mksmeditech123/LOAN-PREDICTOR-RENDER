import streamlit as st
import requests
import pandas as pd
import joblib
import time
import os

# ==============================
# CONFIG
# ==============================
API_URL = "https://loan-api.onrender.com/predict"  # your Render API

st.set_page_config(page_title="Loan Predictor", layout="centered")

# ==============================
# LOAD LOCAL MODEL (SAFE PATH)
# ==============================
@st.cache_resource
def load_local_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "model.pkl")
        columns_path = os.path.join(BASE_DIR, "columns.pkl")

        model = joblib.load(model_path)
        columns = joblib.load(columns_path)

        return model, columns
    except Exception as e:
        print("Model load error:", e)
        return None, None

model, columns = load_local_model()

# ==============================
# UI HEADER
# ==============================
st.title("🏦 AI Loan Approval System")
st.markdown("### ⚡ Fast Prediction (Cloud + Local Fallback)")

# ==============================
# INPUT FORM
# ==============================
with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        app_income = st.number_input("Applicant Income", min_value=0)
        coapp_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_term = st.number_input("Loan Term", min_value=0)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("🚀 Predict")

# ==============================
# PREDICTION LOGIC
# ==============================
if submitted:

    data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    start_time = time.time()

    # ==============================
    # TRY API FIRST
    # ==============================
    with st.spinner("⏳ Trying cloud API (waking server if needed)..."):
        try:
            response = requests.post(API_URL, json=data, timeout=5)

            if response.status_code == 200:
                result = response.json()
                source = "🌐 Cloud API (Render)"
            else:
                raise Exception("API error")

        except Exception as e:
            # ==============================
            # FALLBACK TO LOCAL MODEL
            # ==============================
            st.warning("⚠️ API slow/unavailable → Using LOCAL model ⚡")

            if model is not None:
                df = pd.DataFrame([data])
                df = pd.get_dummies(df)
                df = df.reindex(columns=columns, fill_value=0)

                pred = model.predict(df)[0]

                result = {
                    "prediction": "Approved" if pred == 1 else "Rejected"
                }

                source = "💻 Local Model (Instant)"
            else:
                st.error("❌ Local model not found")
                st.stop()

    end_time = time.time()

    # ==============================
    # RESULT DISPLAY
    # ==============================
    st.markdown("---")

    if result["prediction"] == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"📡 Source: {source}")
    st.write(f"⏱ Response Time: {round(end_time - start_time, 2)} sec")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("🚀 Built with Streamlit + FastAPI + Machine Learning")
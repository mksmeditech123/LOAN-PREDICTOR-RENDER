mport requests
import pandas as pd
impimport streamlit as st
iort joblib
import time

# ==============================
# CONFIG
# ==============================
API_URL = "https://loan-api.onrender.com/predict"  # your API

st.set_page_config(page_title="Loan Predictor", layout="centered")

# ==============================
# LOAD LOCAL MODEL (FAST FALLBACK)
# ==============================
@st.cache_resource
def load_local_model():
    try:
        model = joblib.load("model/model.pkl")
        columns = joblib.load("model/columns.pkl")
        return model, columns
    except:
        return None, None

model, columns = load_local_model()

# ==============================
# UI
# ==============================
st.title("🏦 AI Loan Approval System")
st.markdown("### ⚡ Fast + Smart Prediction (API + Local Backup)")

# INPUTS
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("🚀 Predict"):

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
                source = "🌐 API (Render)"
            else:
                raise Exception("API failed")

        except:
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
                source = "💻 Local Model (Fast)"
            else:
                st.error("❌ No model available")
                st.stop()

    end_time = time.time()

    # ==============================
    # RESULT DISPLAY
    # ==============================
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
st.caption("🚀 Built with Streamlit + FastAPI + ML")
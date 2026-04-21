import streamlit as st
import requests

# 🔗 Your Render API URL
API_URL = "https://loan-api.onrender.com/predict"   # 🔥 replace if different

st.set_page_config(page_title="Loan Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Fill details to check loan approval status")

# 📋 Input Fields
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

# 🚀 Predict Button
if st.button("Predict"):

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

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == "Approved":
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection Error: {e}")
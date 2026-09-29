import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

# Page Configuration
st.set_page_config(
    page_title="AI Health Triage & Supply Chain - India",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Smart Health & Supply Chain Resilience System")
st.caption("Powered by Google AI Gemini & Machine Learning | Designed for India's Public Health Network")

# Load ML Model & CSV Datasets
@st.cache_resource
def load_resources():
    model = joblib.load('disease_model.pkl')
    symptom_df = pd.read_csv('Symptom2Disease.csv')
    bed_df = pd.read_csv('Hospital_Bed_Capacity.csv')
    supply_df = pd.read_csv('Suppy_Chain_Shipment_Data.csv')
    return model, symptom_df, bed_df, supply_df

try:
    disease_model, symptom_df, bed_df, supply_df = load_resources()
    st.sidebar.success("All ML Models & Datasets Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading resources: {e}")

# Safely Check Streamlit Cloud Secrets without crashing locally
api_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Fallback to sidebar text input if no cloud secret is found
if not api_key:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

location = st.sidebar.text_input("Patient Location / District:", value="Salem, Tamil Nadu")
language = st.sidebar.selectbox("Preferred Response Language:", ["English", "Tamil", "Hindi", "Telugu"])

st.subheader("1. Enter Patient Symptoms")
user_input = st.text_area(
    "Describe symptoms in plain text:",
    placeholder="Example: High fever, severe headache, intense pain behind eyes, joint pain, small red spots."
)

if st.button("Run AI Triage & Analysis", type="primary"):
    if not api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar to run the full pipeline.")
    elif not user_input.strip():
        st.warning("Please enter patient symptoms first.")
    else:
        # Predict Disease
        prediction = disease_model.predict([user_input])[0]
        probs = disease_model.predict_proba([user_input])[0]
        confidence = max(probs) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Disease", prediction)
        with col2:
            st.metric("Prediction Confidence", f"{confidence:.1f}%")
        with col3:
            st.metric("Location", location)

        st.divider()
        st.subheader("2. Infrastructure & Supply Chain Readiness")

        bed_col, supply_col = st.columns(2)
        with bed_col:
            st.write("🏥 **Hospital Bed Capacity (Local Network)**")
            st.dataframe(bed_df[['Department', 'Total_Beds', 'Free_Beds', 'Free_ICU_Beds']].head(5))

        with supply_col:
            st.write("📦 **Supply Chain Logistics Monitoring**")
            st.dataframe(supply_df[['product group', 'sub classification', 'shipment mode', 'line item quantity']].head(5))

        st.divider()
        st.subheader("3. Google AI Triage & Action Plan")

        genai.configure(api_key=api_key)

        prompt = f"""
        You are an emergency healthcare triage AI assistant supporting ASHA workers and public health officers across India.

        Patient Location: {location}
        Reported Symptoms: {user_input}
        ML Model Prediction: {prediction} (Confidence: {confidence:.1f}%)
        Response Language: {language}

        Please provide a structured report:
        1. Condition Overview: Explain {prediction} simply for local citizens. Include local terminology (e.g. Tamil terms if applicable).
        2. Immediate Steps for ASHA Worker/Patient: 3 clear, actionable steps.
        3. Resource Allocation & Supply Logistics: Assess if emergency bed allocation or medicine shipment is required.
        """

        with st.spinner("Generating Google AI Guidance..."):
            try:
                gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                response = gemini_model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Gemini API Call Failed: {e}")
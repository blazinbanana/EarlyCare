import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import zipfile


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Liver Disease Detector",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Utility Functions
# ----------------------------
def wrangle(filepath):
    if zipfile.is_zipfile(filepath):
        df = pd.read_csv(filepath, compression='zip', encoding='latin-1')
    else:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df

def make_predictions(data_filepath, model_filepath):
    """Batch predictions for a dataset file."""
    X_test = wrangle(data_filepath)
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    y_test_pred = model.predict(X_test)
    y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="Result")
    return y_test_pred

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'Model/New-Advanced-liver-sickness-detector.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------------------------
# Header
# ----------------------------
st.title("Advanced Liver Disease Detection")
st.subheader("Early Palliative Care Assessment Tool")

st.markdown("""
This tool helps healthcare providers identify patients who may benefit from early palliative care 
interventions based on routine blood test results.
""")

# ----------------------------
# Sidebar Mode Selection
# ----------------------------
mode = st.sidebar.radio(
    "Choose Prediction Mode:",
    ("Single Patient Input", "Batch Prediction (Upload CSV)")
)

# ----------------------------
# SINGLE PATIENT MODE
# ----------------------------
if mode == "Single Patient Input":
    st.sidebar.header("Patient Blood Test Results")

    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.sidebar.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=1.2)
    direct_bilirubin = st.sidebar.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.3)
    alkaline_phosphatase = st.sidebar.number_input("Alkaline Phosphatase (IU/L)", min_value=0, value=200)
    sgpt = st.sidebar.number_input("SGPT (Alamine Aminotransferase) (IU/L)", min_value=0, value=35)
    sgot = st.sidebar.number_input("SGOT (Aspartate Aminotransferase) (IU/L)", min_value=0, value=40)
    total_protein = st.sidebar.number_input("Total Protein (g/dL)", min_value=0.0, value=7.0)
    albumin = st.sidebar.number_input("Albumin (g/dL)", min_value=0.0, value=4.5)
    ag_ratio = st.sidebar.number_input("A/G Ratio", min_value=0.0, value=1.1)

    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame({
            "Age of the patient": [age],
            "Gender of the patient": [gender],  # Pass as string, not 0/1
            "Total Bilirubin": [total_bilirubin],
            "Direct Bilirubin": [direct_bilirubin],
            "Alkphos Alkaline Phosphotase": [alkaline_phosphatase],
            "Sgpt Alamine Aminotransferase": [sgpt],
            "Sgot Aspartate Aminotransferase": [sgot],
            "Total Protiens": [total_protein],
            "ALB Albumin": [albumin],
            "A/G Ratio Albumin and Globulin Ratio": [ag_ratio]
        })


        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.header("Prediction Results")

        if prediction[0] == 1:
            st.error("**High Risk of Advanced Liver Disease**")
            st.write(f"**Probability:** {probability[0][1]:.2%}")
            st.markdown("""
            **Clinical Recommendation:**  
            - Consider palliative care consultation  
            - Discuss goals of care with patient and family  
            - Assess symptom burden and quality of life  
            """)
        else:
            st.success("**Low Risk of Advanced Liver Disease**")
            st.write(f"**Probability:** {probability[0][0]:.2%}")
            st.markdown("""
            **Clinical Recommendation:**  
            - Continue routine monitoring  
            - Maintain standard care pathway  
            """)

# ----------------------------
# BATCH MODE (CSV Upload)

else:
    st.sidebar.header("Batch Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or ZIP", type=["csv", "zip"])

    if uploaded_file is not None:
        # Detect file type dynamically
        if uploaded_file.name.endswith(".zip"):
            temp_path = "temp_uploaded_file.zip"
        else:
            temp_path = "temp_uploaded_file.csv"

        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        model_path = os.path.join(os.path.dirname(__file__), 'Model/New-Advanced-liver-sickness-detector.pkl')

        st.write("Generating predictions...")

        # Pass correct file path depending on upload type
        predictions = make_predictions(temp_path, model_path)

        st.success("Predictions generated successfully!")
        st.dataframe(predictions.head())

        # Option to download results
        csv = predictions.to_csv().encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="liver_predictions.csv",
            mime="text/csv"
        )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:**  
This tool is for decision support purposes only.  
Clinical decisions should be made by qualified healthcare professionals.
""")

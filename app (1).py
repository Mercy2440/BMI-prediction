import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# 1. Load the pre-trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('bmi_predictor_pipeline.joblib')

pipeline = load_pipeline()
model = pipeline['model']
scaler = pipeline['scaler']
trained_features = pipeline['features']

# Helper function for CD4 Grouping
def cd4_group(value):
    s = str(value).strip().lower().replace(',', '')
    if s in ('nan', 'none', ''):
        return '350+'  # Fallback to dataset mode
    nums = re.findall(r'\d+\.?\d*', s)
    n = float(nums[0]) if nums else None
    if '<' in s:
        return '<=200' if (n is None or n <= 200) else '350+'
    if '>' in s:
        if n is None or n <= 200:
            return '>200 (value not given)'
        return '350+' if n >= 350 else '201-349'
    if n is None:
        return '350+'
    if n <= 200:
        return '<=200'
    return '201-349' if n < 350 else '350+'

# Set page styling
st.set_page_config(page_title="BMI Category Predictor", layout="centered")
st.title("🩺 HIV Client BMI Category Predictor")
st.write("Enter patient details below to predict their likely BMI Category using the optimized Gradient Boosting model.")

# App layout: forms for inputs
with st.form("prediction_form"):
    st.subheader("Demographics & Social Factors")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", ["F", "M"])
        marital_status = st.selectbox("Marital Status", ["yes", "no"])
        age = st.number_input("Age at Reporting", min_value=18, max_value=120, value=40)
    with col2:
        family_members = st.selectbox("Family Members", ["yes", "no"])
        employment = st.selectbox("Employment", ["yes", "no"])
        ahd_client = st.selectbox("AHD Client Status", ["Yes", "No"])

    st.subheader("Clinical Parameters")
    col3, col4 = st.columns(2)
    with col3:
        months_prescription = st.number_input("Months Of Prescription", min_value=0, max_value=60, value=5)
        last_risk_score = st.number_input("Last Risk Score", min_value=0.0, max_value=1.0, value=0.003, format="%.6f")
    with col4:
        baseline_cd4 = st.text_input("Baseline CD4 Result (e.g., 350, <=200, >200)", "350")
        latest_cd4 = st.text_input("Latest CD4 Result (e.g., 450, <=200, >200)", "450")

    submit_button = st.form_submit_button("Predict BMI Category")

if submit_button:
    # 2. Process custom CD4 inputs
    b_cd4_group = cd4_group(baseline_cd4)
    l_cd4_group = cd4_group(latest_cd4)

    # 3. Create DataFrame representing user input
    input_dict = {
        'Sex': sex,
        'Marital status': marital_status,
        'Family Members': family_members,
        'Employmet': employment,
        'AHD Client': ahd_client,
        'Months Of Prescription': months_prescription,
        'Age at Reporting': age,
        'Last Risk Score': last_risk_score,
        'Baseline CD4 group': b_cd4_group,
        'Latest CD4 group': l_cd4_group
    }
    
    input_df = pd.DataFrame([input_dict])

    # 4. Perform One-Hot Encoding and categorical alignment
    categorical_cols = ['Sex', 'Marital status', 'Family Members', 'Employmet', 'AHD Client', 'Baseline CD4 group', 'Latest CD4 group']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
        
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Re-align features to match the exact order of the columns used in model training
    for col in trained_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[trained_features]

    # 5. Scale the numerical columns
    numeric_cols = ['Months Of Prescription', 'Age at Reporting', 'Last Risk Score']
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    # 6. Execute prediction
    prediction = model.predict(input_encoded)[0]
    probabilities = model.predict_proba(input_encoded)[0]
    classes = model.classes_

    # 7. Render prediction outputs
    st.success(f"🎯 **Predicted BMI Category: {prediction}**")
    
    st.write("**Class Probability Distributions:**")
    prob_df = pd.DataFrame({"Category": classes, "Probability": probabilities})
    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
    st.dataframe(prob_df)

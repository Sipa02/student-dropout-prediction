import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model

# Load model
model = joblib.load('best_rf_model.joblib')

st.title("Student Status Prediction App")

# Form input
with st.form("prediction_form"):
    admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, step=0.1)
    age_at_enrollment = st.number_input("Age at enrollment", min_value=15, max_value=100, step=1)

    # Academic data
    first_enrolled = st.number_input("1st sem enrolled", min_value=1, step=1)
    first_eval = st.number_input("1st sem evaluations", min_value=0, step=1)
    first_approved = st.number_input("1st sem approved", min_value=0, step=1)
    first_no_eval = st.number_input("1st sem without evaluations", min_value=0, step=1)

    second_enrolled = st.number_input("2nd sem enrolled", min_value=1, step=1)
    second_approved = st.number_input("2nd sem approved", min_value=0, step=1)

    marital_status = st.selectbox("Marital status", [1, 2, 3, 4, 5])
    application_mode = st.selectbox("Application mode", list(range(1, 61)))
    attendance = st.selectbox("Daytime/evening attendance", [1, 0])  # 1 = Day, 0 = Evening
    gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male
    international = st.selectbox("International", [0, 1])
    scholarship = st.selectbox("Scholarship holder", [0, 1])
    tuition_up_to_date = st.selectbox("Tuition fees up to date", [0, 1])
    debtor = st.selectbox("Debtor", [0, 1])

    # Parent education
    mother_edu = st.number_input("Mother's qualification", min_value=1, max_value=44, step=1)
    father_edu = st.number_input("Father's qualification", min_value=1, max_value=44, step=1)

    # Target (untuk training/testing, bisa diabaikan saat produksi)
    # encoder_target = joblib.load('label_encoder_target.joblib')
    # target = st.selectbox("Target (for testing only)", encoder_target.classes_.tolist())

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_data = {
            "Admission grade": admission_grade,
            "Age at enrollment": age_at_enrollment,
            "Curricular units 1st sem (enrolled)": first_enrolled,
            "Curricular units 1st sem (evaluations)": first_eval,
            "Curricular units 1st sem (approved)": first_approved,
            # "Curricular units 1st sem (grade)":  
            "Curricular units 1st sem (without evaluations)": first_no_eval,
            "Curricular units 2nd sem (enrolled)": second_enrolled,
            "Curricular units 2nd sem (approved)": second_approved,
            "Marital status": marital_status,
            "Application mode": application_mode,
            "Daytime/evening attendance": attendance,  # removed \t
            "Gender": gender,
            "International": international,
            "Scholarship holder": scholarship,
            "Tuition fees up to date": tuition_up_to_date,
            "Debtor": debtor,
            "Mother's qualification": mother_edu,
            "Father's qualification": father_edu,
            "Target": target
        }

        input_df = pd.DataFrame([input_data])
        # Rename column to match model expectation
        input_df.rename(columns={"Daytime/evening attendance": "Daytime/evening attendance\t"}, inplace=True)

        processed_df = preprocess_for_model(input_df)
        prediction = model.predict(processed_df)

        decoded_prediction = encoder_target.inverse_transform(prediction)
        st.success(f"🎯 Predicted Student Status: **{decoded_prediction[0]}**")

    except Exception as e:
        st.error(f"🚨 Terjadi error saat memproses prediksi: {str(e)}")

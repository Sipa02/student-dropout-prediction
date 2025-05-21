import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model
from mapping_dicts import FIELD_MAPPINGS

# Load model
model = joblib.load('best_rf_model.joblib')

st.title("🎓 Student Status Prediction App")

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

    # Pilihan dengan mapping
    marital_display = st.selectbox("Marital status", list(FIELD_MAPPINGS["Marital status"].keys()))
    marital_status = FIELD_MAPPINGS["Marital status"][marital_display]

    application_mode_display = st.selectbox("Application mode", list(FIELD_MAPPINGS["Application mode"].keys()))
    application_mode = FIELD_MAPPINGS["Application mode"][application_mode_display]

    attendance_display = st.selectbox("Daytime/evening attendance", list(FIELD_MAPPINGS["Daytime/evening attendance\t"].keys()))
    attendance = FIELD_MAPPINGS["Daytime/evening attendance\t"][attendance_display]

    gender_display = st.selectbox("Gender", list(FIELD_MAPPINGS["Gender"].keys()))
    gender = FIELD_MAPPINGS["Gender"][gender_display]

    international_display = st.selectbox("International", list(FIELD_MAPPINGS["International"].keys()))
    international = FIELD_MAPPINGS["International"][international_display]

    scholarship_display = st.selectbox("Scholarship holder", list(FIELD_MAPPINGS["Scholarship holder"].keys()))
    scholarship = FIELD_MAPPINGS["Scholarship holder"][scholarship_display]

    tuition_display = st.selectbox("Tuition fees up to date", list(FIELD_MAPPINGS["Tuition fees up to date"].keys()))
    tuition_up_to_date = FIELD_MAPPINGS["Tuition fees up to date"][tuition_display]

    debtor_display = st.selectbox("Debtor", list(FIELD_MAPPINGS["Debtor"].keys()))
    debtor = FIELD_MAPPINGS["Debtor"][debtor_display]

    # Parent education
    mother_edu_display = st.selectbox("Mother's qualification", list(FIELD_MAPPINGS["Mother's qualification"].keys()))
    mother_edu = FIELD_MAPPINGS["Mother's qualification"][mother_edu_display]
    father_edu_display = st.selectbox("Father's qualification", list(FIELD_MAPPINGS["Father's qualification"].keys()))
    father_edu = FIELD_MAPPINGS["Father's qualification"][mother_edu_display]

    # Target (for testing only)
    encoder_target = joblib.load('label_encoder_target.joblib')
    target = st.selectbox("Target (for testing only)", encoder_target.classes_.tolist())

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_data = {
            "Admission grade": admission_grade,
            "Age at enrollment": age_at_enrollment,
            "Curricular units 1st sem (enrolled)": first_enrolled,
            "Curricular units 1st sem (evaluations)": first_eval,
            "Curricular units 1st sem (approved)": first_approved,
            "Curricular units 1st sem (without evaluations)": first_no_eval,
            "Curricular units 2nd sem (enrolled)": second_enrolled,
            "Curricular units 2nd sem (approved)": second_approved,
            "Marital status": marital_status,
            "Application mode": application_mode,
            "Daytime/evening attendance\t": attendance,
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
        processed_df = preprocess_for_model(input_df)
        prediction = model.predict(processed_df)

        status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

        if st.button("Make Prediction"):
            prediction = model.predict(input_scaled)
            predicted_class = int(prediction[0])
            prediction_label = status_mapping.get(predicted_class, "Unknown")
            proba = model.predict_proba(input_scaled)[0]

            st.subheader(f"🎯 Predicted Status: {prediction_label}")
            st.write(f"Confidence: {np.max(proba) * 100:.2f}%")


     

    except Exception as e:
        st.error(f"🚨 Terjadi error saat memproses prediksi: {str(e)}")

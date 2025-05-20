import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model

# Load model dan encoder target
model = joblib.load("best_rf_model.joblib")
encoder_target = joblib.load("encoder_target.joblib")

# Set UI
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")

st.subheader("Masukkan Data Mahasiswa")

# Form input manual
application_mode = st.number_input("Application Mode (angka)", min_value=1)
mother_edu = st.number_input("Mother's Qualification (angka)", min_value=1)
father_edu = st.number_input("Father's Qualification (angka)", min_value=1)

# Data akademik dan fitur tambahan (kamu bisa sesuaikan)
admission_grade = st.slider("Admission Grade", 0.0, 200.0, step=0.1)
age_at_enrollment = st.number_input("Age at Enrollment", min_value=15, max_value=100)
marital_status = st.selectbox("Marital Status", [1, 2])
attendance = st.selectbox("Attendance", [1, 0])  # Daytime/evening
gender = st.selectbox("Gender", [1, 0])
international = st.selectbox("International", [1, 0])
scholarship = st.selectbox("Scholarship Holder", [1, 0])
tuition_paid = st.selectbox("Tuition Fees Up To Date", [1, 0])
debtor = st.selectbox("Debtor", [1, 0])
# Semester info
enrolled_1st = st.number_input("Enrolled 1st Sem", min_value=1)
eval_1st = st.number_input("Evaluations 1st Sem", min_value=0)
approved_1st = st.number_input("Approved 1st Sem", min_value=0)
grade_1st = st.number_input("Grade 1st Sem", min_value=0.0)
no_eval_1st = st.number_input("No Evaluations 1st Sem", min_value=0)
enrolled_2nd = st.number_input("Enrolled 2nd Sem", min_value=1)
approved_2nd = st.number_input("Approved 2nd Sem", min_value=0)

# Prediksi
if st.button("Prediksi Status"):
    try:
        # Gabung jadi dataframe
        input_df = pd.DataFrame([{
            "Application mode": application_mode,
            "Mother's qualification": mother_edu,
            "Father's qualification": father_edu,
            "Admission grade": admission_grade,
            "Age at enrollment": age_at_enrollment,
            "Marital status": marital_status,
            "Daytime/evening attendance\t": attendance,
            "Gender": gender,
            "International": international,
            "Scholarship holder": scholarship,
            "Tuition fees up to date": tuition_paid,
            "Debtor": debtor,
            "Curricular units 1st sem (enrolled)": enrolled_1st,
            "Curricular units 1st sem (evaluations)": eval_1st,
            "Curricular units 1st sem (approved)": approved_1st,
            "Curricular units 1st sem (grade)": grade_1st,
            "Curricular units 1st sem (without evaluations)": no_eval_1st,
            "Curricular units 2nd sem (enrolled)": enrolled_2nd,
            "Curricular units 2nd sem (approved)": approved_2nd,
            "Target": 0  # dummy aja biar preprocessing gak error
        }])

        df_processed = preprocess_for_model(input_df)

        pred = model.predict(df_processed)
        label = encoder_target.inverse_transform(pred)[0]

        st.success(f"✅ Status Mahasiswa: **{label}**")
    except Exception as e:
        st.error(f"❌ Error saat memproses prediksi: {e}")

import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model

# Load model dan encoder
model = joblib.load("best_rf_model.joblib")
encoder_target = joblib.load("encoder_target.joblib")
encoder_app_group = joblib.load("label_encoder_app_group.joblib")
encoder_mother_edu = joblib.load("label_encoder_mother_edu.joblib")
encoder_father_edu = joblib.load("label_encoder_father_edu.joblib")

# Title
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")

st.subheader("Masukkan Data Mahasiswa")

# Form input satuan
application_mode = st.selectbox("Application Mode", encoder_app_group.classes_)
mother_edu = st.selectbox("Mother's Qualification", encoder_mother_edu.classes_)
father_edu = st.selectbox("Father's Qualification", encoder_father_edu.classes_)

if st.button("Prediksi Status"):
    try:
        # Buat dataframe dari input
        input_df = pd.DataFrame([{
            "Application_mode": application_mode,
            "Mother_qualification": mother_edu,
            "Father_qualification": father_edu
        }])

        # Preprocessing
        df_processed = preprocess_for_model(
            input_df,
            le_app_group=encoder_app_group,
            le_mother_edu=encoder_mother_edu,
            le_father_edu=encoder_father_edu,
            is_training=False
        )

        # Prediksi
        prediction = model.predict(df_processed)
        pred_label = encoder_target.inverse_transform(prediction)[0]

        st.success(f"✅ Prediksi Status Mahasiswa: **{pred_label}**")

    except Exception as e:
        st.error(f"Terjadi error saat memproses prediksi: {e}")

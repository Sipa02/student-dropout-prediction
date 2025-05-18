import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model

# Load model dan encoder
model = joblib.load("best_rf_model.joblib")
encoder_target = joblib.load("encoder_target.joblib")
encoder_app_group = joblib.load("encoder_app_group.joblib")
encoder_mother_edu = joblib.load("encoder_mother_edu.joblib")
encoder_father_edu = joblib.load("encoder_father_edu.joblib")

# Inject encoder yang sudah di-load ke preprocessing
def preprocess_with_loaded_encoders(df):
    df_processed = preprocess_for_model(
        df,
        le_target=encoder_target,
        le_app_group=encoder_app_group,
        le_mother_edu=encoder_mother_edu,
        le_father_edu=encoder_father_edu,
        is_training=False
    )
    return df_processed

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")

uploaded_file = st.file_uploader("Upload file CSV data mahasiswa", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        st.subheader("Data Awal")
        st.dataframe(df_raw.head())

        df_processed = preprocess_with_loaded_encoders(df_raw)
        predictions = model.predict(df_processed)
        predictions_label = encoder_target.inverse_transform(predictions)

        df_raw["Predicted Status"] = predictions_label

        st.subheader("Hasil Prediksi")
        st.dataframe(df_raw[["Predicted Status"]])

        csv_download = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download hasil prediksi CSV",
            data=csv_download,
            file_name="hasil_prediksi.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
else:
    st.info("Silakan upload file .csv yang sesuai format.")


import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_for_model

# Load model dan encoder
model = joblib.load("best_rf_model.joblib")
encoder_target = joblib.load("encoder_target.joblib")
encoder_app_group = joblib.load("encoder_app_group.joblib")
encoder_mother_edu = joblib.load("encoder_mother_edu.joblib")
encoder_father_edu = joblib.load("encoder_father_edu.joblib")

# Inject encoder yang sudah di-load ke preprocessing
def preprocess_with_loaded_encoders(df):
    df_processed = preprocess_for_model(
        df,
        le_target=encoder_target,
        le_app_group=encoder_app_group,
        le_mother_edu=encoder_mother_edu,
        le_father_edu=encoder_father_edu,
        is_training=False
    )
    return df_processed

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")

uploaded_file = st.file_uploader("Upload file CSV data mahasiswa", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        st.subheader("Data Awal")
        st.dataframe(df_raw.head())

        df_processed = preprocess_with_loaded_encoders(df_raw)
        predictions = model.predict(df_processed)
        predictions_label = encoder_target.inverse_transform(predictions)

        df_raw["Predicted Status"] = predictions_label

        st.subheader("Hasil Prediksi")
        st.dataframe(df_raw[["Predicted Status"]])

        csv_download = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download hasil prediksi CSV",
            data=csv_download,
            file_name="hasil_prediksi.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
else:
    st.info("Silakan upload file .csv yang sesuai format.")



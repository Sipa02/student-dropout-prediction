import pandas as pd
import joblib

# Load model dan encoder target
model = joblib.load("best_rf_model.joblib")
target_encoder = joblib.load("encoder_target.joblib")
label_encoders = joblib.load("label_encoders.pkl")  # kalau ada fitur kategorikal

# Fungsi prediksi
def predict_from_dict(input_dict):
    df = pd.DataFrame([input_dict])
    
    # Transformasi label encoding untuk fitur kategorikal
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Pastikan urutan kolom sesuai dengan saat training
    X_train_columns = model.feature_names_in_
    for col in X_train_columns:
        if col not in df.columns:
            df[col] = 0  # isi default kalau kolom kosong

    df = df[X_train_columns]

    # Prediksi
    prediction = model.predict(df)
    final_result = target_encoder.inverse_transform(prediction)[0]
    return final_result

# Contoh input
if __name__ == "__main__":
    input_data = {
        "Usia": 24,
        "Jenis_Kelamin": "Female",
        "Beasiswa": "No",
        "SPP_Lunas": "No",
        "Nilai_Ujian_Masuk": 2.5,
        "Total_Mata_Kuliah_Lulus": 6,
        # tambahkan semua fitur lainnya...
    }

    result = predict_from_dict(input_data)
    print("Prediksi:", result)

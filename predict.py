import joblib
model = joblib.load('best_rf_model.joblib')
result_target = joblib.load('encoder_target.joblib')
def predictions(data):
    if data.shape[0] != 1:
        raise ValueError("Predictions function only supports one row of input.")
    result = model.predict(data)
    return result_target.inverse_transform(result)[0]

### app.py ###

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data, create_sequences
from src.predict_temperature import predict_temperature_for_date

app = Flask(__name__)

# Load model and data at startup to avoid reloading on every request
model_path = 'model/weather_lstm_model.h5'
file_path = 'data/export2016_2024.xlsx'

print("Loading model and data...")
model = tf.keras.models.load_model(model_path)
data, scaler, target_scalers = load_and_preprocess_data(file_path)

SEQ_LENGTH = 60
features = [col for col in data.columns if col != 'date']
print("Model and data loaded successfully!")


@app.route('/predict', methods=['GET'])
def predict_temperature():
    try:
        # Get the date from request parameters
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({'error': 'Date parameter is required.'}), 400

        # Parse the date
        prediction_date = datetime.strptime(
            date_str, '%Y-%m-%d').strftime('%Y-%m-%d')

        # Predict temperature for the given date
        prediction = predict_temperature_for_date(
            model, data, target_scalers, prediction_date, SEQ_LENGTH, features
        )

        # Return the prediction as JSON
        return jsonify({
            'date': prediction_date,
            'predicted_avg_temp': round(prediction[0], 2),
            'predicted_min_temp': round(prediction[1], 2),
            'predicted_max_temp': round(prediction[2], 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

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
        today = datetime.now().strftime('%Y-%m-%d')

        # Ensure the entered date is in the future
        if datetime.strptime(prediction_date, '%Y-%m-%d') <= datetime.strptime(today, '%Y-%m-%d'):
            return jsonify({'error': 'The date must be in the future.'}), 400

        # Predict temperature for the given date
        prediction = predict_temperature_for_date(
            model, data, target_scalers, prediction_date, SEQ_LENGTH, features
        )

        # Extract and inverse transform data from today to the entered date
        historical_data = data[(data['date'] >= pd.to_datetime(today)) &
                               (data['date'] <= pd.to_datetime(prediction_date))]

        # Inverse transform the scaled historical data to get the real values
        historical_data_real = historical_data[['tavg', 'tmin', 'tmax']].copy()
        for feature in ['tavg', 'tmin', 'tmax']:
            historical_data_real[feature] = target_scalers[feature].inverse_transform(
                historical_data_real[feature].values.reshape(-1, 1)
            )

        # Add the date back to the historical data
        historical_data_real['date'] = historical_data['date'].dt.strftime(
            '%Y-%m-%d')

        # Return the prediction and historical data as JSON
        return jsonify({
            'date': prediction_date,
            'predicted_avg_temp': round(prediction[0], 2),
            'predicted_min_temp': round(prediction[1], 2),
            'predicted_max_temp': round(prediction[2], 2),
            'historical_data': historical_data_real.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

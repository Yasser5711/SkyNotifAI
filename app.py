# app.py
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data_preprocessing import load_and_preprocess_data
from main import prepare_input_sequence

app = Flask(__name__)

# Load the trained model
model_path = 'model/weather_lstm_model.h5'
model = tf.keras.models.load_model(model_path)

# Load and preprocess the data
file_path = 'data/export_2023.xlsx'
data, scaler, target_scalers = load_and_preprocess_data(file_path)

SEQ_LENGTH = 30
features = [col for col in data.columns if col != 'date']
num_features = len(features)


def predict_future_temperatures(model, data, start_date, end_date, seq_length, features, target_scalers, batch_size=7):
    current_date = start_date
    current_sequence = prepare_input_sequence(data, seq_length, features)
    predictions = []

    while current_date <= end_date:
        num_days = (end_date - current_date).days + 1
        if num_days > batch_size:
            num_days = batch_size

        input_sequences = np.zeros((num_days, seq_length, len(features)))

        for i in range(num_days):
            input_sequences[i] = current_sequence

        predictions_batch = model.predict(input_sequences)

        for i in range(num_days):
            # Inverse transform the prediction
            inv_prediction = np.zeros(predictions_batch[i].shape)
            for j, target in enumerate(['tavg', 'tmin', 'tmax']):
                inv_prediction[j] = target_scalers[target].inverse_transform(
                    predictions_batch[i, j].reshape(-1, 1)).flatten()

            predictions.append(
                (current_date.strftime('%Y-%m-%d'), inv_prediction))

            # Update the sequence with the new prediction
            new_row = np.zeros((1, len(features)))
            new_row[0, :3] = predictions_batch[i]
            # Keep the non-temperature features the same
            new_row[0, 3:] = current_sequence[-1, 3:]
            current_sequence = np.vstack([current_sequence[1:], new_row])

            # Move to the next day
            current_date += timedelta(days=1)

    return predictions


@app.route('/predict', methods=['GET'])
def predict():
    day = int(request.args.get('day'))
    month = int(request.args.get('month'))
    year = int(request.args.get('year'))
    input_date = datetime(year, month, day)

    # Ensure the input date is in the future and within the reasonable prediction range
    last_data_date = data['date'].max()
    if input_date <= last_data_date:
        return jsonify({'error': 'The date must be in the future.'}), 400

    # Predict temperatures for the specified future date
    predictions = predict_future_temperatures(
        model, data, last_data_date + timedelta(days=1), input_date, SEQ_LENGTH, features, target_scalers)

    # Find the prediction for the requested date
    for date, temps in predictions:
        if date == input_date.strftime('%Y-%m-%d'):
            return jsonify({
                'date': date,
                'avg_temp': temps[0],
                'min_temp': temps[1],
                'max_temp': temps[2]
            })

    return jsonify({'error': 'Prediction not found.'}), 404


if __name__ == '__main__':
    app.run(debug=True)

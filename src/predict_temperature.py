import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_preprocessing import load_and_preprocess_data, create_sequences
from model_evaluation import predict_tomorrow


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_temperature_for_date(model, data, target_scalers, date, seq_length, features):
    # Ensure the date is in datetime format
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    # Calculate how many days into the future the date is
    last_data_date = data['date'].max()
    days_into_future = (date - last_data_date).days

    if days_into_future < 0:
        raise ValueError("The date must be in the future.")

    # Create the initial sequence from the most recent data
    current_sequence = data.iloc[-seq_length:
                                 ][features].values.astype(np.float32)

    # Generate predictions up to the desired date
    for _ in range(days_into_future):
        input_sequence = current_sequence.reshape(1, seq_length, len(features))
        prediction = model.predict(input_sequence)

        # Update the sequence with the new prediction
        new_row = np.zeros((1, len(features)))
        # Use the model's predictions for temperature
        new_row[0, :3] = prediction[0]
        # Copy non-temperature features
        new_row[0, 3:] = current_sequence[-1, 3:]
        current_sequence = np.vstack([current_sequence[1:], new_row])

    # Inverse transform the prediction to get the actual values
    inv_prediction = np.zeros(prediction.shape)
    for i, target in enumerate(['tavg', 'tmin', 'tmax']):
        inv_prediction[:, i] = target_scalers[target].inverse_transform(
            prediction[:, i].reshape(-1, 1)).flatten()

    return inv_prediction[0]


if __name__ == "__main__":
    model_path = 'model/weather_lstm_model.h5'
    file_path = 'data/export2016_2024.xlsx'

    # Load and preprocess the data
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    # Load the trained model
    model = load_model(model_path)

    # Define the date for prediction
    prediction_date = (datetime.now() + timedelta(weeks=20)
                       ).strftime('%Y-%m-%d')  # Predicting for tomorrow

    # Define the sequence length and features
    SEQ_LENGTH = 60
    features = [col for col in data.columns if col != 'date']

    # Predict the temperature
    prediction = predict_temperature_for_date(
        model, data, target_scalers, prediction_date, SEQ_LENGTH, features)

    print(
        f'Predicted temperatures for {prediction_date}: Avg: {prediction[0]:.2f}, Min: {prediction[1]:.2f}, Max: {prediction[2]:.2f}')

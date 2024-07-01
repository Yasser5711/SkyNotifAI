# main.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data, create_sequences
from src.model_training import build_and_train_model
from src.model_evaluation import evaluate_model, predict_tomorrow
import tensorflow as tf


def get_date_input():
    day = int(input("Enter the day (1-31): "))
    month = int(input("Enter the month (1-12): "))
    year = int(input("Enter the year (e.g., 2024): "))
    return datetime(year, month, day)


def prepare_input_sequence(data, seq_length, features):
    # Get the last sequence from the data
    input_sequence = data.iloc[-seq_length:
                               ][features].values.astype(np.float32)
    return input_sequence


def predict_future_temperatures(model, data, start_date, end_date, seq_length, features, target_scalers):
    current_date = start_date
    current_sequence = prepare_input_sequence(data, seq_length, features)
    predictions = []

    while current_date <= end_date:
        # Reshape the sequence to match the model input
        input_sequence = current_sequence.reshape(1, seq_length, len(features))

        # Predict the next day's temperatures
        prediction = model.predict(input_sequence)

        # Inverse transform the prediction
        inv_prediction = np.zeros(prediction.shape)
        for i, target in enumerate(['tavg', 'tmin', 'tmax']):
            inv_prediction[:, i] = target_scalers[target].inverse_transform(
                prediction[:, i].reshape(-1, 1)).flatten()

        predictions.append(
            (current_date.strftime('%Y-%m-%d'), inv_prediction[0]))

        # Update the sequence with the new prediction
        new_row = np.zeros((1, len(features)))
        new_row[0, :3] = prediction[0]
        # Keep the non-temperature features the same
        new_row[0, 3:] = current_sequence[-1, 3:]
        current_sequence = np.vstack([current_sequence[1:], new_row])

        # Move to the next day
        current_date += timedelta(days=1)

    return predictions


def main():
    # Load and preprocess the data
    file_path = 'data/export_2023.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    # Print the range of dates available in the dataset
    print(f"Data ranges from {data['date'].min()} to {data['date'].max()}")

    # Parameters
    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'date']
    num_features = len(features)
    num_outputs = 3  # Number of target variables (tavg, tmin, tmax)

    # Create sequences for training
    X, y = create_sequences(data, SEQ_LENGTH, features)

    # Build and train the model
    model, X_val, y_val, history = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, num_outputs, epochs=200, learning_rate=0.0005)

    # Save the model
    model_path = 'model/weather_lstm_model.h5'
    model.save(model_path)
    print("Model trained and saved.")

    # Evaluate the model
    evaluate_model(model_path, X_val, y_val, target_scalers, features)

    # Get date input from the user
    input_date = get_date_input()

    # Ensure the input date is in the future and within the reasonable prediction range
    last_data_date = data['date'].max()
    if input_date <= last_data_date:
        print(
            f"The date {input_date.strftime('%Y-%m-%d')} is not in the future.")
        return

    # Predict temperatures for the specified future date
    predictions = predict_future_temperatures(
        model, data, last_data_date + timedelta(days=1), input_date, SEQ_LENGTH, features, target_scalers)

    for date, temps in predictions:
        print(
            f'Predicted temperatures for {date}: Avg: {temps[0]:.2f}, Min: {temps[1]:.2f}, Max: {temps[2]:.2f}')


if __name__ == "__main__":
    main()

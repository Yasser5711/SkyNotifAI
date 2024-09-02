import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data, create_sequences
from src.model_training import build_and_train_model, plot_learning_curves, predict_future_temperatures_and_plot
from src.model_evaluation import evaluate_model
import tensorflow as tf


def main():
    file_path = 'data/export2016_2024.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    print(f"Data ranges from {data['date'].min()} to {data['date'].max()}")

    SEQ_LENGTH = 60  # Experiment with different sequence lengths
    features = [col for col in data.columns if col != 'date']
    num_features = len(features)
    num_outputs = 3

    X, y = create_sequences(data, SEQ_LENGTH, features)

    model, X_val, y_val, history = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, num_outputs, epochs=150, learning_rate=0.0005)

    model_path = 'model/weather_lstm_model.h5'
    model.save(model_path)
    print("Model trained and saved.")
    plot_learning_curves(history)

    evaluate_model(model_path, X_val, y_val, target_scalers, features)

    # Predict temperatures for the next month
    start_date = data['date'].max() + timedelta(days=1)
    end_date = start_date + timedelta(days=30)
    predictions = predict_future_temperatures_and_plot(
        model, data, start_date, end_date, SEQ_LENGTH, features, target_scalers)

    for date, temps in predictions:
        print(
            f'Predicted temperatures for {date}: Avg: {temps[0]:.2f}, Min: {temps[1]:.2f}, Max: {temps[2]:.2f}')


if __name__ == "__main__":
    main()

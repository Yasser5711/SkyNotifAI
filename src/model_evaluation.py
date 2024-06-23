# src/model_evaluation.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os


def evaluate_model(model_path, X_test, y_test, target_scalers, features):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions and y_test to the original scale
    inv_predictions = np.zeros(predictions.shape)
    inv_y_test = np.zeros(y_test.shape)

    for i, target in enumerate(['tavg', 'tmin', 'tmax']):
        inv_predictions[:, i] = target_scalers[target].inverse_transform(
            predictions[:, i].reshape(-1, 1)).flatten()
        inv_y_test[:, i] = target_scalers[target].inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()

    # Print the first prediction vs actual
    print(f'Predicted: {inv_predictions[0]}, Actual: {inv_y_test[0]}')

    # Create a DataFrame to save predictions
    results = pd.DataFrame(inv_predictions, columns=[
                           f'Predicted_{feat}' for feat in features[:3]])
    results[['Actual_tavg', 'Actual_tmin', 'Actual_tmax']] = inv_y_test

    # Save the DataFrame to an Excel file
    os.makedirs('predictions', exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'predictions/predictions_{current_time}.xlsx'
    results.to_excel(file_name, index=False)
    print(f"Predictions saved to {file_name}")


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data, create_sequences

    # Load and preprocess the data
    file_path = '../data/weather_data.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    # Parameters
    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'date']
    num_features = len(features)

    # Create sequences
    X, y = create_sequences(data, SEQ_LENGTH, features)

    # Load test data
    from model_training import build_and_train_model
    model, X_test, y_test = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, 3)

    # Evaluate the model
    evaluate_model('../model/weather_lstm_model.h5',
                   X_test, y_test, target_scalers, features)

# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_excel(file_path)

    # Select features to normalize (excluding the date column)
    features = ['tavg', 'tmin', 'tmax']

    # Ensure 'date' is parsed as datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Remove columns with all NaN values
    data = data.dropna(axis=1, how='all')

    # Check for missing values and fill them
    data[features] = data[features].fillna(data[features].mean())

    # Add month and season as features
    data['Month'] = data['date'].dt.month
    data['Season'] = data['date'].dt.month % 12 // 3 + 1

    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features + ['Month', 'Season']])

    # Convert scaled data back to DataFrame for easier manipulation
    data_scaled = pd.DataFrame(
        data_scaled, columns=features + ['Month', 'Season'])
    data_scaled['date'] = data['date']

    # Fit separate scalers for target variables
    target_scalers = {}
    for target in features:
        target_scalers[target] = MinMaxScaler()
        data_scaled[target] = target_scalers[target].fit_transform(
            data[[target]])

    return data_scaled, scaler, target_scalers


def create_sequences(data, seq_length, features):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i+seq_length][features].values.astype(np.float32)
        # Only target tavg, tmin, tmax
        y = data.iloc[i+seq_length][features[:3]].values.astype(np.float32)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    file_path = '../data/export.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)
    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'date']
    X, y = create_sequences(data, SEQ_LENGTH, features)
    print(X.shape, y.shape)

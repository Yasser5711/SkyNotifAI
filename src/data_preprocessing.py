import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_excel(file_path)

    # Select features to normalize (excluding the date column)
    features = ['tavg', 'tmin', 'tmax', 'prcp',
                'wdir', 'wspd', 'wpgt', 'pres']

    # Remove columns with all NaN values
    data = data.dropna(axis=1, how='all')

    # Check for missing values and fill them
    data[features] = data[features].fillna(data[features].mean())

    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Convert scaled data back to DataFrame for easier manipulation
    data_scaled = pd.DataFrame(data_scaled, columns=features)
    data_scaled['date'] = data['date']  # Add the date column back

    # Fit separate scalers for target variables
    target_scalers = {}
    for target in ['tavg', 'tmin', 'tmax']:
        target_scalers[target] = MinMaxScaler()
        data_scaled[target] = target_scalers[target].fit_transform(
            data[[target]])

    return data_scaled, scaler, target_scalers


def create_sequences(data, seq_length, features):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i+seq_length][features].values.astype(np.float32)
        # Assuming the first 3 features are tavg, tmin, tmax
        y = data.iloc[i+seq_length][features[:3]].values.astype(np.float32)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    file_path = '../data/export.xlsx'
    data, scaler = load_and_preprocess_data(file_path)
    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'date']
    X, y = create_sequences(data, SEQ_LENGTH, features)
    print(X.shape, y.shape)

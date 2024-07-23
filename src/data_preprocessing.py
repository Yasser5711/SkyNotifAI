import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import numpy as np


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_excel(file_path)

    features = ['tavg', 'tmin', 'tmax', 'prcp', 'pres', 'wspd', 'wdir']
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(axis=1, how='all')

    # Use KNN imputer for missing values
    imputer = KNNImputer(n_neighbors=5)
    data[features] = imputer.fit_transform(data[features])

    # Add seasonal features
    data['Month'] = data['date'].dt.month
    data['DayOfYear'] = data['date'].dt.dayofyear
    data['WeekOfYear'] = data['date'].dt.isocalendar().week
    data['DayOfWeek'] = data['date'].dt.dayofweek
    data['Season'] = data['date'].dt.month % 12 // 3 + 1
    data['Sin_DayOfYear'] = np.sin(2 * np.pi * data['DayOfYear'] / 365.0)
    data['Cos_DayOfYear'] = np.cos(2 * np.pi * data['DayOfYear'] / 365.0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(
        data[features + ['Month', 'DayOfYear', 'WeekOfYear', 'DayOfWeek', 'Season', 'Sin_DayOfYear', 'Cos_DayOfYear']])

    data_scaled = pd.DataFrame(data_scaled, columns=features + [
                               'Month', 'DayOfYear', 'WeekOfYear', 'DayOfWeek', 'Season', 'Sin_DayOfYear', 'Cos_DayOfYear'])
    data_scaled['date'] = data['date']

    target_scalers = {}
    for target in features[:3]:
        target_scalers[target] = MinMaxScaler()
        data_scaled[target] = target_scalers[target].fit_transform(
            data[[target]])

    return data_scaled, scaler, target_scalers


def create_sequences(data, seq_length, features):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i + seq_length][features].values.astype(np.float32)
        y = data.iloc[i + seq_length][features[:3]].values.astype(np.float32)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    file_path = '../data/export.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)
    SEQ_LENGTH = 60  # Experiment with different sequence lengths
    features = [col for col in data.columns if col != 'date']
    X, y = create_sequences(data, SEQ_LENGTH, features)
    print(X.shape, y.shape)

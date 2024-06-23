import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def build_and_train_model(X, y, seq_length, num_features, num_outputs, batch_size=1, epochs=10):
    # Split the data into training and testing sets
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True,
                             input_shape=(seq_length, num_features)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(num_outputs)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, X_test, y_test


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data, create_sequences

    # Load and preprocess the data
    file_path = '../data/export.xlsx'
    data, scaler = load_and_preprocess_data(file_path)

    # Parameters
    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'date']
    num_features = len(features)
    num_outputs = 3  # Number of target variables (tavg, tmin, tmax)

    # Create sequences
    X, y = create_sequences(data, SEQ_LENGTH, features)

    # Build and train the model
    model, X_test, y_test = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, num_outputs)

    # Save the model
    model.save('../model/weather_lstm_model.h5')
    print("Model trained and saved.")

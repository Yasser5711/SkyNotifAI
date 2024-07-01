# src/model_training.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from .data_preprocessing import create_sequences, load_and_preprocess_data


def build_and_train_model(X, y, seq_length, num_features, num_outputs, batch_size=32, epochs=100, learning_rate=0.001):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True,
                             input_shape=(seq_length, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100, return_sequences=False),
        tf.keras.layers.Dense(
            50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(num_outputs)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

    return model, X_val, y_val, history


if __name__ == "__main__":
    file_path = '../data/export_2023.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    SEQ_LENGTH = 30
    features = [col for col in data.columns if col != 'Date']
    num_features = len(features)
    num_outputs = 3  # Number of target variables (tavg, tmin, tmax)

    X, y = create_sequences(data, SEQ_LENGTH, features)

    model, X_val, y_val, history = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, num_outputs)

    model.save('../model/weather_lstm_model.h5')
    print("Model trained and saved.")

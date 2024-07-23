import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from .data_preprocessing import create_sequences, load_and_preprocess_data
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd


def build_and_train_model(X, y, seq_length, num_features, num_outputs, batch_size=32, epochs=100, learning_rate=0.001):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(
            seq_length, num_features)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=False)),
        tf.keras.layers.Dense(
            64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(num_outputs)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, lr_scheduler])

    return model, X_val, y_val, history


def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('Mean Absolute Error')
    plt.legend()

    plt.show()


def predict_future_temperatures_and_plot(model, data, start_date, end_date, seq_length, features, target_scalers):
    current_date = start_date
    current_sequence = data.iloc[-seq_length:
                                 ][features].values.astype(np.float32)
    predictions = []

    while current_date <= end_date:
        input_sequence = current_sequence.reshape(1, seq_length, len(features))
        prediction = model.predict(input_sequence)

        inv_prediction = np.zeros(prediction.shape)
        for i, target in enumerate(['tavg', 'tmin', 'tmax']):
            inv_prediction[:, i] = target_scalers[target].inverse_transform(
                prediction[:, i].reshape(-1, 1)).flatten()

        predictions.append(
            (current_date.strftime('%Y-%m-%d'), inv_prediction[0]))

        # Update the sequence with the new prediction
        new_row = np.zeros((1, len(features)))
        # Use the model's predictions for temperature
        new_row[0, :3] = prediction[0]
        # Copy non-temperature features
        new_row[0, 3:] = current_sequence[-1, 3:]
        current_sequence = np.vstack([current_sequence[1:], new_row])

        current_date += timedelta(days=1)

    # Convert predictions to DataFrame
    prediction_dates = [date for date, _ in predictions]
    prediction_values = [values for _, values in predictions]
    prediction_df = pd.DataFrame(prediction_values, columns=[
                                 'Predicted_tavg', 'Predicted_tmin', 'Predicted_tmax'], index=prediction_dates)

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df.index,
             prediction_df['Predicted_tavg'], label='Predicted Avg Temp')
    plt.plot(prediction_df.index,
             prediction_df['Predicted_tmin'], label='Predicted Min Temp')
    plt.plot(prediction_df.index,
             prediction_df['Predicted_tmax'], label='Predicted Max Temp')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Predictions for the Next Month')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    return predictions


if __name__ == "__main__":
    file_path = '../data/export.xlsx'
    data, scaler, target_scalers = load_and_preprocess_data(file_path)

    SEQ_LENGTH = 60  # Experiment with different sequence lengths
    features = [col for col in data.columns if col != 'date']
    num_features = len(features)
    num_outputs = 3  # Number of target variables (tavg, tmin, tmax)

    X, y = create_sequences(data, SEQ_LENGTH, features)

    model, X_val, y_val, history = build_and_train_model(
        X, y, SEQ_LENGTH, num_features, num_outputs, epochs=150, learning_rate=0.0005)

    model.save('../model/weather_lstm_model.h5')
    print("Model trained and saved.")
    plot_learning_curves(history)

    # Predict temperatures for the next month
    start_date = data['date'].max() + timedelta(days=1)
    end_date = start_date + timedelta(days=30)
    predict_future_temperatures_and_plot(
        model, data, start_date, end_date, SEQ_LENGTH, features, target_scalers)

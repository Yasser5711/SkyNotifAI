# main.py
from src.data_preprocessing import load_and_preprocess_data, create_sequences
from src.model_training import build_and_train_model
from src.model_evaluation import evaluate_model

# File path to the data
file_path = 'data/export.xlsx'

# Preprocess the data
data, scaler, target_scalers = load_and_preprocess_data(file_path)

# Create sequences
SEQ_LENGTH = 30
features = ['tavg', 'tmin', 'tmax', 'prcp',
            'wdir', 'wspd', 'wpgt', 'pres']
X, y = create_sequences(data, SEQ_LENGTH, features)

num_features = len(features)
num_outputs = 3  # Number of target variables (tavg, tmin, tmax)
# Build and train the model
model, X_test, y_test = build_and_train_model(
    X, y, SEQ_LENGTH, num_features, num_outputs)

# Save the model
model.save('model/weather_lstm_model.h5')

# Evaluate the model
evaluate_model('model/weather_lstm_model.h5', X_test,
               y_test, target_scalers, features)

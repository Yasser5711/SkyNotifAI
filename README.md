# 🌦️ SkyNotifAI

This project is a **SkyNotifAI** built with Flask, TensorFlow, and Scikit-Learn. It allows you to predict the average, minimum, and maximum temperatures for a future date using a pre-trained machine learning model. Additionally, it provides historical temperature data leading up to the specified prediction date.

## 📂 Project Structure

```bash
project_directory/
│
├── app.py               # 🌐 Flask API server
├── README.md            # 📃 Project documentation
├── src/
│   ├── data_preprocessing.py  # 🧹 Data preprocessing utilities
│   ├── model_training.py      # 🏋️ Model building and training
│   ├── model_evaluation.py    # 🧪 Model evaluation functions
│   ├── predict_temperature.py # 🔮 Prediction utilities
│
├── model/
│   └── weather_lstm_model.h5  # 🗂️ Trained LSTM model file
│
└── data/
    └── export2016_2024.xlsx   # 📊 Historical weather data
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone  https://github.com/Yasser5711/SkyNotifAI.git
cd IA
```

### 2. Install Dependencies

Ensure you have Python 3.7+ and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Flask API

Start the Flask server with the following command:

```bash
python app.py
```

The API will be running at http://127.0.0.1:5000.

## 📊 Usage

### 1. Predict Future Temperatures

You can make a GET request to the /predict endpoint with a date parameter in YYYY-MM-DD format. This will return the predicted average, minimum, and maximum temperatures for the specified date along with historical data.

Example Request with curl:

```bash
curl "http://127.0.0.1:5000/predict?date=2024-08-30"
```

Example Response:

```json
{
    "date": "2024-12-01",
    "predicted_avg_temp": 15.67,
    "predicted_min_temp": 10.23,
    "predicted_max_temp": 20.56,
    "historical_data": [
        {"date": "2024-08-20", "tavg": 18.3, "tmin": 12.1, "tmax": 24.5},
        {"date": "2024-08-21", "tavg": 17.9, "tmin": 11.8, "tmax": 23.4},
        ...
    ]
}
```

### 2. Making Requests with Python

You can also make requests using Python's requests library:

```python
import requests
response = requests.get("http://127.0.0.1:5000/predict?date=2024-08-30")
print(response.json())
```

## 🛠️ Customization

### Adjusting the Model

The model is built and trained using the code in src/model_training.py. You can adjust hyperparameters like sequence length, learning rate, number of epochs, etc., to experiment with the model's performance.

### Modifying Data Preprocessing

If you need to preprocess data differently, you can modify src/data_preprocessing.py. This file handles tasks like loading data, imputing missing values, scaling features, and creating time-based features.

## 🔍 Project Details

### Model Architecture

The model used in this project is a Bidirectional LSTM combined with Conv1D layers to capture sequential patterns in the weather data. The architecture is designed to predict the average, minimum, and maximum temperatures for a future date based on historical weather data.

### Data Preprocessing

Data preprocessing involves:

- Handling Missing Data: Imputation using KNN.
- Feature Scaling: Scaling features to a 0-1 range using MinMaxScaler.
- Feature Engineering: Adding seasonal features (e.g., month, day of the year).

### Prediction Logic

The prediction logic can be found in src/predict_temperature.py, where the model predicts the future temperature based on a sequence of historical data points.

### Instructions to Use:

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install the necessary Python packages.
2. **Run the API**: Use `python app.py` to start the Flask API server.
3. **Make Requests**: Use tools like `curl`, Postman, or Python's `requests` library to interact with the API.

### Additional Tips:

- **Experiment**: Try adjusting model parameters in `src/model_training.py` to see how it impacts predictions.
- **Error Handling**: The API already has basic error handling for missing dates or invalid inputs, but consider adding more detailed validation depending on your use case.

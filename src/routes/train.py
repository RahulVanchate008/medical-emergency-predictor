from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = FastAPI()

# Paths to save the model and scaler
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"

# Define request model for training
class TrainRequest(BaseModel):
    data_path: str  # Path to CSV file with dataset

def preprocess_data(data_path: str):
    # Load data
    data = pd.read_csv(data_path)
    
    # Extract heart rate column and reshape for LSTM input
    heart_rate = data['heart_rate'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    heart_rate_scaled = scaler.fit_transform(heart_rate)

    # Save the scaler for later use with new data
    joblib.dump(scaler, SCALER_PATH)
    
    # Prepare sequences for LSTM model (using a window of 60, for example)
    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(heart_rate_scaled)):
        X.append(heart_rate_scaled[i-sequence_length:i, 0])
        y.append(heart_rate_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM input (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm_model(input_shape):
    # Define LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Preprocess data
        X, y = preprocess_data(request.data_path)
        
        # Build and train LSTM model
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, batch_size=32, epochs=10, verbose=1)
        
        # Save the model
        model.save(MODEL_PATH)
        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_model(test_data_path: str):
    try:
        # Load trained model and scaler
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise HTTPException(status_code=404, detail="Model or scaler not found. Train the model first.")
        
        model = Sequential()
        model.load_weights(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Preprocess test data
        test_data = pd.read_csv(test_data_path)
        heart_rate = test_data['heart_rate'].values.reshape(-1, 1)
        heart_rate_scaled = scaler.transform(heart_rate)
        
        # Prepare sequences for LSTM prediction
        X_test = []
        sequence_length = 60
        for i in range(sequence_length, len(heart_rate_scaled)):
            X_test.append(heart_rate_scaled[i-sequence_length:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)  # Reverse scaling for interpretability

        # Return predictions as a list
        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import io

router = APIRouter()

MODEL_PATH = "lstm_mit_bih_model.h5"
SCALER_PATH = "scaler.pkl"

def preprocess_data(file: UploadFile):
    df = pd.read_csv(io.BytesIO(file.file.read()))
    heart_rate = df['heart_rate'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    heart_rate_scaled = scaler.fit_transform(heart_rate)

    joblib.dump(scaler, SCALER_PATH)

    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(heart_rate_scaled)):
        X.append(heart_rate_scaled[i-sequence_length:i, 0])
        y.append(heart_rate_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@router.post("/train-LSTM")
async def train_model(file: UploadFile = File(...)):
    try:
        X, y = preprocess_data(file)

        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, batch_size=32, epochs=10, verbose=1)

        model.save(MODEL_PATH)
        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test Endpoint
@router.post("/test-LSTM")
async def test_model(file: UploadFile = File(...)):
    try:
        # Load trained model and scaler
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise HTTPException(status_code=404, detail="Model or scaler not found. Train the model first.")
        
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Load test data from uploaded file
        df = pd.read_csv(io.BytesIO(file.file.read()))
        heart_rate_values = df['heart_rate'].values.reshape(-1, 1)
        
        # Scale heart rate values
        heart_rate_scaled = scaler.transform(heart_rate_values)
        
        # Prepare sequences for testing (ensure sequence length of 60)
        X_test = []
        sequence_length = 60
        if len(heart_rate_scaled) < sequence_length:
            raise HTTPException(status_code=400, detail="Not enough data. Provide at least 60 heart rate values.")
        
        X_test.append(heart_rate_scaled[-sequence_length:, 0])  # Use the last 60 values
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict the next heart rate value
        prediction = model.predict(X_test)
        
        # Convert prediction to float for JSON serialization
        predicted_heart_rate = float(scaler.inverse_transform(prediction).flatten()[0])
        
        # Return prediction
        return {"predicted_heart_rate": predicted_heart_rate}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

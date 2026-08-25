from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import io

app = FastAPI(title="TensorFlow-LMS API", description="Sales Forecasting using LSTM")

# Global variables
model = None
scaler = MinMaxScaler(feature_range=(0,1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

@app.get("/")
def home():
    return {"message": "TensorFlow-LMS API is running. Go to /docs to test"}

@app.post("/train/")
async def train_model(file: UploadFile = File(...)):
    global model

    # 1. Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Scale data
    data = df[['Sales']].values
    scaled_data = scaler.fit_transform(data)

    # 3. Create sequences
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 4. Train Test Split
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Build LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 6. Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # 7. Evaluate
    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    rmse = np.sqrt(mean_squared_error(y_test_inv, pred))

    return JSONResponse(content={"status": "Model Trained", "RMSE": float(rmse)})

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...), days: int = 30):
    global model
    if model is None:
        return {"error": "Please train model first using /train endpoint"}

    # 1. Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Scale
    data = df[['Sales']].values
    scaled_data = scaler.transform(data)

    # 3. Take last 60 days
    seq_length = 60
    last_sequence = scaled_data[-seq_length:]
    forecast_list = []

    # 4. Predict next N days
    for _ in range(days):
        X_pred = last_sequence.reshape(1, seq_length, 1)
        pred = model.predict(X_pred, verbose=0)
        forecast_list.append(pred[0,0])
        last_sequence = np.append(last_sequence[1:], [[pred[0,0]]], axis=0)

    # 5. Inverse transform
    forecast_inv = scaler.inverse_transform(np.array(forecast_list).reshape(-1,1))

    # 6. Create dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)

    result = pd.DataFrame({"Date": future_dates, "Predicted_Sales": forecast_inv.flatten()})

    return JSONResponse(content=result.to_dict(orient="records"))
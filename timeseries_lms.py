import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
st.title("📈 LMS Sales Forecasting - Time Series with TensorFlow")
st.write("Project: TensorFlow-LMS-TimeSeries-Forecaster")

# ================== STEP 1: DATA LOAD ==================
st.header("Step 1: Data Load")
uploaded_file = st.file_uploader("sales.csv upload karo", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data:", df.head())

    # TUMHARE DATA KE HISAB SE: sirf date aur sum_total rakho
    df = df[['date', 'sum_total']] 
    df.columns = ['Date', 'Sales'] # naam easy kar diye

    # Date theek karo + NaN delete
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    
    # 1 din ki total sales nikal lo - Time Series ke liye zaruri
    df = df.groupby('Date')['Sales'].sum().reset_index()
    df = df.sort_values('Date')

    st.write("Processed Data:", df.head())
    st.line_chart(df.set_index('Date'))

    # ================== STEP 2: DATA PREPROCESSING ==================
    st.header("Step 2: Data Preprocessing")

    # 1. Normalize data 0-1 ke beech
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Sales'].values.reshape(-1,1))

    # 2. Create sequences for time series. 10 din dekh ke next din predict
    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 3. Train Test Split
    split = int(len(X) * 0.8) # 80% train, 20% test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # ================== STEP 3: NEURAL NETWORK ==================
    st.header("Step 3: Neural Network Architecture")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    st.text(model.summary())

    # ================== STEP 4: MODEL COMPILE ==================
    st.header("Step 4: Model Compile")
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.success("Model compiled with Adam optimizer and MSE loss")

    # ================== STEP 5: MODEL TRAIN ==================
    st.header("Step 5: Model Training")
    history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test), verbose=1)
    st.success("Training Complete!")

    # Graph for loss
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title('Model Loss')
    ax.legend()
    st.pyplot(fig)

    # ================== STEP 6: MODEL EVALUATION ==================
    st.header("Step 6: Model Evaluation")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # RMSE calculate
    rmse = np.sqrt(np.mean(((predictions - y_test_actual)**2)))
    st.metric("RMSE", f"{rmse:.2f}")

    # Plot Actual vs Predicted
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(y_test_actual, label='Actual Sales')
    ax2.plot(predictions, label='Predicted Sales')
    ax2.set_title('Actual vs Predicted Sales')
    ax2.legend()
    st.pyplot(fig2)

    # ================== STEP 7: MODEL PREDICTION ==================
    st.header("Step 7: Future Prediction")
    last_10_days = scaled_data[-seq_length:]
    future_pred = []

    for _ in range(30): # next 30 days predict
        x_input = last_10_days.reshape(1, seq_length, 1)
        yhat = model.predict(x_input, verbose=0)
        future_pred.append(yhat[0,0])
        last_10_days = np.append(last_10_days[1:], yhat)

    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1,1))
    st.write("Next 5 Days Predicted Sales:")
    st.write(future_pred[:5])

    # ================== STEP 8: MODEL SAVE ==================
    st.header("Step 8: Model Save")
    model.save('lms_sales_model.h5')
    st.success("Model saved as lms_sales_model.h5")
    with open('lms_sales_model.h5', 'rb') as f:
        st.download_button("Download Model", data=f, file_name='lms_sales_model.h5')

else:
    st.warning("Pehle sales.csv file upload karo")
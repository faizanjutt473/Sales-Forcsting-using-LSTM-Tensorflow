# TensorFlow-LMS: Sales Forecasting
LSTM based sales forecasting using TensorFlow, Streamlit and FastAPI.

## Features
- Upload sales.csv and forecast next 30 days
- Streamlit UI for demo
- FastAPI backend for integration
- Trained LSTM Model

## How to Run
### Streamlit App
pip install -r requirements.txt
streamlit run app_streamlit.py

### FastAPI
uvicorn app_fastapi:app --reload
Open: http://127.0.0.1:8000/docs

# Sales Forecasting using LSTM

'''
Aaj kal har business ke liye future ki sales predict karna bohat zaruri hai. Is project ka maqsad hai sales.csv data ko use karke agle 30 din ki sales predict karna.
Humne iske liye TensorFlow + Keras ka LSTM model banaya hai.
LSTM ek tarah ka Deep Learning model hai jo time series data ke "pattern" aur "trend" ko yaad rakhta hai.Jaise: Eid pe sales zyada, Sunday ko kam, last week achi gayi to is week bhi achi jayegi.App Streamlit pe banayi hai taake koi bhi bina coding ke sirf sales.csv upload kare aur foran forecast le.

'''
#overview



'''
Data Upload: User sales.csv file upload kar sakta hai
Data Preprocessing: Date ko sort karna, Missing values fill karna
LSTM Model: 60 din ka data le kar agla din predict karta hai
Training: Model 20 epochs tak train hota hai
Evaluation: RMSE se accuracy check hoti hai
Visualization: Actual vs Predicted ka graph banta hai
Future Forecast: Agle 30 din ki sales predict karke download bhi kar sakte ho
Download Model: Trained model ko .h5 file me save kar sakte ho

'''


# Technologies Used:


#Python 3.10   programming language
#TensorFlow 2.15  LSTM bnaany ky liay
#Streamlit     web app UI bnanay ky liay
#Pandas + Numpy  data  handle krny ky liay
#Matplotlib    data handle krnny ky liay
#Scikit-learn    remse and data scaling 



#How it Works - Flow
'''
Upload → sales.csv upload karo jisme Date, Sales columns hon
Process → Data ko 0-1 me scale karte hain
Train → LSTM model 60 din ka sequence dekh kar seekhta hai
Predict → Last 60 din se agle 30 din predict karta hai
Output → RMSE, Graph aur forecast_30_days.csv file

'''


# Folder Structure
'''

TensorFlow-LMS/
├── timeseries_lms.py   # Main Streamlit App
├── sales.csv           # Input Data
├── requirements.txt    # Libraries
└── lstm_sales_model.h5 # Trained Model - Auto save hota hai


'''

#Future Improvements:


'''
Multiple products ki forecasting
Holiday/Sale events ko feature me add karna
Model ko auto retrain karna

'''

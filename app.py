import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import yfinance as yf

TF_ENABLE_ONEDNN_OPTS=0

START = '2020-01-01'
current = datetime.today().strftime('%Y-%m-%d')

# Most Watched by Yahoo Finance Users
stock_list = ["AAPL", "NVDA", "MSFT", "GOOG", "AMZN", "META", "TSLA", "WMT",
              "JPM", "XOM", "HD", "PG", "JNJ", "BAC", "KO", "NFLX", "AMD",
              "BABA", "CSCO", "MCD", "IBM", "GE", "VZ", "DIS", "PFE", "T",
              "C", "BA", "INTC", "F"]

def get_data(ticker):
    data = yf.download(ticker, START, current)
    return data

st.title("LSTM Stock Price Forecasting App")

st.subheader('Stock information')
ticker = st.selectbox("Please choose a stock:", stock_list)

load_text = st.text('Loading Data...')
show_data = get_data(ticker)
load_text.text('Data extracted successfully.')

st.write(f"Dataset of {ticker} Stock:")
st.dataframe(show_data)

show_data['Move_Avg(30D)'] = show_data['Close'].rolling(30).mean()
show_data['Move_Avg(90D)'] = show_data['Close'].rolling(90).mean()

def conversion(selected_time):
    convert_num = {
        '1 week': 7,
        '1 month': 30,
        '3 months': 90,
        '1 year': 365,
    }
    
    return convert_num[selected_time]

time_ticker = st.selectbox("Please select the period of time for the stock trend you want to view:", ['1 week', '1 month', '3 months', '1 year'])
days_to_show = conversion(time_ticker)

new_data = show_data.iloc[-days_to_show :]

# plot line chart
line_data = new_data[['Close', 'Move_Avg(30D)', 'Move_Avg(90D)']]
st.write(f"Stock Trend for the Last {time_ticker}")

latest_close_price = show_data.iloc[-1]["Close"]
latest_mov30 = show_data.iloc[-1]['Move_Avg(30D)']
latest_mov90 = show_data.iloc[-1]['Move_Avg(90D)']

col1,col2,col3 = st.columns(3)
with col1:
    st.metric('Current Close Price', f'${latest_close_price:.2f}')
with col2:
    st.metric('Current Moving Average (30D)', f'${latest_mov30:.2f}')
with col3:
    st.metric('Current Moving Average (90D)', f'${latest_mov90:.2f}')

st.line_chart(line_data)    

# Open Lstm model and do forecasting
import pickle
pickle_path = r"C:\Users\Edwin\Python\bootcamp\Projects\lstm\model\lstm.pkl"
pickle_scaler_path = r"C:\Users\Edwin\Python\bootcamp\Projects\lstm\model\scaler.pkl"

with open(pickle_path, 'rb') as file:
    saved_model = pickle.load(file)

with open(pickle_scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Prepare the data for LSTM forecasting  
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Prepare the dataset for prediction  
data_close = show_data['Close'].values  
data_close = data_close.reshape(-1, 1)
data_scaled = scaler.transform(data_close)

def make_predictions(prediction_days):
    # Define the time step for LSTM (you might want to keep this constant)
    time_step = 100  # Using last 100 days for prediction

    # Create dataset for LSTM predictions using the latest available data  
    X, Y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Make predictions using the LSTM model  
    predicted_price_scaled = saved_model.predict(X)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Prepare predicted prices for plotting  
    predicted_dates = pd.date_range(start = new_data.index[-1] + pd.Timedelta(days = 1), periods = prediction_days, freq = 'B')
    predicted_df = pd.DataFrame(data = predicted_price[-prediction_days:], index = predicted_dates, columns = ['Predicted Close'])

    return predicted_df

def price_difference(latest_close_price, predicted_df):
    return ((predicted_df['Predicted Close'].iloc[-1] - latest_close_price) / latest_close_price) * 100

st.subheader('Stock Closing Price Prediction')
prediction_days = st.slider("How many days would you like to predict?", 1, 365)
predicted_df = make_predictions(prediction_days)

price_diff = price_difference(latest_close_price, predicted_df)

# Combine actual and predicted prices for visualization  
combined_df = pd.concat([show_data.iloc[-365 :][['Close']], predicted_df], axis = 1)
combined_df.columns = ['Actual Close', 'Predicted Close']

col4, col5 = st.columns(2)
with col4:
    st.metric('Predicted price change', f'{price_diff:.2f}%')
with col5:
    st.metric('Predicted Close Price', f'${predicted_df["Predicted Close"].iloc[-1]:.2f}')

# Plot the actual vs predicted prices  
st.line_chart(combined_df)

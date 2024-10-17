import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

TF_ENABLE_ONEDNN_OPTS=0

START = '2020-01-01'
current = datetime.today().strftime('%Y-%m-%d')

def get_data(name):
    data = yf.download(name, START, current)
    return data

# Use GOOGL for training the model
df = get_data("GOOGL")
df = df.reset_index()

# Split the train dataset and test dataset
train_df = pd.DataFrame(df['Close'][0: int(len(df) * 0.80)])
test_df = pd.DataFrame(df['Close'][int(len(df) * 0.80):int(len(df))])
train_df.shape # (964, 1)
test_df.shape # (241, 1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_df)

# Create x_train and y_train; prediction of 100 days
x_train = []
y_train = []

for i in range(100, train_df.shape[0]):
    x_train.append(training_set_scaled[i-100: i])    
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape # (864, 100, 1)
y_train.shape # (864, )

# Create the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

regressor = Sequential()

# first layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# fourth layer
regressor.add(LSTM(units = 50))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 50, batch_size = 32)

# Save the model
import pickle
with open('./model/lstm.pkl', 'wb') as lstm_file:
    pickle.dump(regressor, lstm_file)

with open('./model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)
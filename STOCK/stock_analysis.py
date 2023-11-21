import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(user_input):
    yf.pdr_override() 
    start = "2013-01-01"
    end = "2023-12-31"
    
    try:
        df = pdr.get_data_yahoo(user_input, start=start, end=end)
        return df
    except Exception as e:
        return "Not available"

def plot_stock_data(df):
    fig= plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    plt.legend()
    plt.show()
def plot_stock_data_with_100MA(df):
    
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    plt.legend()
    plt.show()

def plot_stock_data_with_100MA_200MA(df):
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(ma100, 'r', label="100 MA")
    plt.plot(ma200, 'g', label='200 MA')
    plt.plot(df.Close, 'b', label='Original Price')
    
    plt.legend()  # Add legend
    
    plt.show()

def prepare_predictions(model, scaler, df):
    data_train = pd.DataFrame(df['Close'][:int(len(df) * 0.70)])
    data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_array = scaler.fit_transform(data_train)

    past_100_days = data_train.tail(100)
    final_df = pd.concat([past_100_days, data_test], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    return y_test, y_predicted


def plot_predictions(y_test, y_predicted):
    
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def get_latest_price(user_input):
    yf.pdr_override() 
    df = pdr.get_data_yahoo(user_input)
    return df['Close'].iloc[-1]

model = tf.keras.models.load_model(r'STOCK/keras_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

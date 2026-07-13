import streamlit as st
import math
import pandas_datareader as web
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

st.title('BTC Price Forecasting with LSTM')
st.write('Made by Samara Acosta')
st.info('This app builds a Long-Short Term Mmeory RNN for BTC price forecasting!')

# Create CSV with historical data

import yfinance as yf
end_date =  datetime.today().strftime('%Y-%m-%d')

btc = yf.download(
    "BTC-USD",
    start="2021-06-10",
    end = end_date
)

btc.to_csv('bitcoin_historical_data.csv')
btc.fillna(inplace=True)
df = pd.read_csv('bitcoin_historical_data.csv')
df

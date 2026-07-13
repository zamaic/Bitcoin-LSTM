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
import yfinance as yf

# Introduction

st.title('BTC Price Forecasting with LSTM')
st.write('Made by Samara Acosta')
st.info('This app builds a Long-Short Term Mmeory RNN for BTC price forecasting!')

# Create CSV with historical data

end_date =  datetime.today().strftime('%Y-%m-%d')
btc = yf.download(
    "BTC-USD",
    start="2021-06-10",
    end = end_date
)

btc.columns = btc.columns.droplevel(1)
btc.reset_index(inplace=True)
btc.columns = 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'

btc.ffill(inplace=True)
btc.to_csv("bitcoin_historical_data.csv", index=False)

with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('bitcoin_historical_data.csv')
    df

    st.write('**Features (X)**')
    st.write('Last 60 days (Open, High, Low, Close)')
    X_display = df[['Date', 'Open', 'High', 'Low', 'Close']].tail(60)
    X_display
    
    st.write('**Target (y)**')
    st.write('Close price of next day')
    y_display = df[['Date', 'Close']].tail(60)
    y_display


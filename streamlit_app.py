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

end_date = datetime.today().strftime('%Y-%m-%d')

btc = yf.download(
    "BTC-USD",
    start="2021-06-10",
    end=end_date,
    auto_adjust=False
)

# Convert yfinance MultiIndex to normal columns
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc.reset_index(inplace=True)

# Select only the columns that actually exist
columns_needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

btc = btc[[col for col in columns_needed if col in btc.columns]]

# Fill missing data
btc.ffill(inplace=True)

# Save historical data
btc.to_csv("bitcoin_historical_data.csv", index=False)


# Data

with st.expander('Data', expanded=True):

    st.write('**Raw Data**')

    df = pd.read_csv('bitcoin_historical_data.csv')

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

    st.write('**Features (X)**')

    st.write('Last 60 days (Open, High, Low, Close)')

    feature_columns = [
        col for col in ['Date', 'Open', 'High', 'Low', 'Close']
        if col in df.columns
    ]

    X_display = df[feature_columns].tail(60)

    st.dataframe(
        X_display,
        use_container_width=True,
        height=400
    )

    st.write('**Target (y)**')

    st.write('Close price of next day')

    y_display = df[
        [col for col in ['Date', 'Close'] if col in df.columns]
    ].tail(60)

    st.dataframe(
        y_display,
        use_container_width=True,
        height=400
    )


# Data Visualization

with st.expander('Data Visualization', expanded=True):

    fig, ax = plt.subplots(figsize=(16, 8))

    price_columns = [
        col for col in ['Open', 'High', 'Low', 'Close']
        if col in btc.columns
    ]

    ax.plot(btc[price_columns])

    ax.set_title(
        'Price Bitcoin - OHLC',
        fontsize=24
    )

    ax.set_xlabel(
        'Date',
        fontsize=18
    )

    ax.set_ylabel(
        'Price USD',
        fontsize=18
    )

    ax.legend(price_columns)

    ax.grid(True)

    st.pyplot(fig)

    st.write('**Close Price**')

    if 'Close' in btc.columns:
        st.line_chart(
            btc.set_index('Date')['Close']
        )

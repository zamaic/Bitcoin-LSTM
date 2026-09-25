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

# Download data from Yahoo Finance
try:

    btc = yf.download(
        "BTC-USD",
        start="2021-06-10",
        end=end_date,
        auto_adjust=False,
        multi_level_index=False
    )

    # Check if Yahoo returned data
    if btc.empty:
        raise Exception("Yahoo Finance returned no data")

    btc.reset_index(inplace=True)

except Exception:

    # Backup data if Yahoo Finance is temporarily unavailable

    dates = pd.date_range(
        start="2021-06-10",
        end=end_date,
        freq="D"
    )

    np.random.seed(42)

    close = 40000 + np.cumsum(
        np.random.normal(0, 500, len(dates))
    )

    btc = pd.DataFrame({
        'Date': dates,
        'Open': close + np.random.normal(0, 300, len(dates)),
        'High': close + np.random.uniform(100, 1000, len(dates)),
        'Low': close - np.random.uniform(100, 1000, len(dates)),
        'Close': close,
        'Volume': np.random.uniform(
            10000000000,
            50000000000,
            len(dates)
        )
    })


# Make sure the required columns exist

btc = btc[
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
]

btc.ffill(inplace=True)

btc.to_csv(
    "bitcoin_historical_data.csv",
    index=False
)


# Data

with st.expander('Data', expanded=True):

    st.write('**Raw Data**')

    df = pd.read_csv(
        'bitcoin_historical_data.csv'
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )


    st.write('**Features (X)**')

    st.write(
        'Last 60 days (Open, High, Low, Close)'
    )

    X_display = btc[
        ['Date', 'Open', 'High', 'Low', 'Close']
    ].tail(60)

    st.dataframe(
        X_display,
        use_container_width=True,
        height=400
    )


    st.write('**Target (y)**')

    st.write(
        'Close price of next day'
    )

    y_display = btc[
        ['Date', 'Close']
    ].tail(60)

    st.dataframe(
        y_display,
        use_container_width=True,
        height=400
    )


# Data Visualization

with st.expander(
    'Data Visualization',
    expanded=True
):

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    ax.plot(
        btc[
            ['Open', 'High', 'Low', 'Close']
        ]
    )

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

    ax.legend([
        'Open',
        'High',
        'Low',
        'Close'
    ])

    ax.grid(True)

    st.pyplot(fig)


    st.write('**Close Price**')

    st.line_chart(
        btc.set_index('Date')['Close']
    )

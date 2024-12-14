import matplotlib
matplotlib.use('Agg')  # Use the non-GUI Agg backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data (Replace 'data.csv' with your file name)
data = pd.read_csv("AAPL_historical_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Function to calculate RSI
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

# Function to calculate Moving Averages
def calculate_moving_average(close, period):
    return close.rolling(window=period).mean()

# Apply indicators
data['RSI'] = calculate_rsi(data['Close'], period=14)
data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'], period=20, std_dev=2)
data['SMA_20'] = calculate_moving_average(data['Close'], period=20)
data['SMA_200'] = calculate_moving_average(data['Close'], period=200)

# Save the processed data
data.to_csv("AAPL_historical_data.csv", index=False)

# Plot example: RSI and Bollinger Bands
plt.figure(figsize=(14, 7))

# Plot Bollinger Bands with Close Price
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Close'], label="Close Price", color='blue')
plt.plot(data['Date'], data['BB_Upper'], label="BB Upper", color='red')
plt.plot(data['Date'], data['BB_Middle'], label="BB Middle", color='green')
plt.plot(data['Date'], data['BB_Lower'], label="BB Lower", color='red')
plt.title("Bollinger Bands")
plt.legend()

# Plot RSI
plt.subplot(2, 1, 2)
plt.plot(data['Date'], data['RSI'], label="RSI", color='purple')
plt.axhline(70, color='red', linestyle='--', label="Overbought")
plt.axhline(30, color='green', linestyle='--', label="Oversold")
plt.title("RSI")
plt.legend()

plt.tight_layout()

# Save the plot to a file
plt.savefig("output_plot.png")  # Save the plot as an image file

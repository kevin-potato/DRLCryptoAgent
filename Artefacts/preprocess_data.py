import pandas as pd
import numpy as np

def EMA(x, n):
    alpha = 2 / (n + 1)
    ema = [x[0]]
    for i in range(1, len(x)):
        ema.append(alpha * x[i] + (1 - alpha) * ema[i-1])
    return ema

def MACD(close, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = EMA(close, fast_period)
    ema_slow = EMA(close, slow_period)
    dif = np.array(ema_fast) - np.array(ema_slow)
    dea = EMA(dif, signal_period)
    macd = dif - np.array(dea)
    return dif, dea, macd

def RSI(close, rsi_period=30):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def KDJ(df, period=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    rsv = 100 * ((df['close'] - low_min) / (high_max - low_min))
    k = rsv.ewm(span=k_period, adjust=False).mean()
    d = k.ewm(span=d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def preprocess_data(btc_data, short_window=5, long_window=10):
    btc_data['Short_MA'] = btc_data['close'].rolling(window=short_window, min_periods=1).mean()
    btc_data['Long_MA'] = btc_data['close'].rolling(window=long_window, min_periods=1).mean()
    btc_data['MA_30'] = btc_data['close'].rolling(window=30, min_periods=1).mean()
    btc_data['MA_90'] = btc_data['close'].rolling(window=90, min_periods=1).mean()
    btc_data['DIF'], btc_data['DEA'], btc_data['MACD'] = MACD(btc_data['close'], fast_period=12, slow_period=26, signal_period=9)
    btc_data['RSI'] = RSI(btc_data['close'], rsi_period=14)
    btc_data['K'], btc_data['D'], btc_data['J'] = KDJ(btc_data)
    return btc_data

if __name__ == "__main__":
    btc_data = pd.read_csv('BTC-USD.csv')
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data.set_index('date', inplace=True)

    btc_data = preprocess_data(btc_data)
    btc_data.fillna(0, inplace=True)
    btc_data.to_csv('BTC_TECH.csv')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import PPO, DQN
from train_model import StockEnv

def generate_signals_s1(data):
    data['Signal_s1'] = 0
    data['Signal_s1'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position_s1'] = data['Signal_s1'].diff()
    return data

def generate_signals_s2(data):
    buy_signal = ((data['MACD'] > data['DEA']) & (data['RSI'] < 70) & (data['close'] > data['Short_MA']))
    sell_signal = ((data['MACD'] < data['DEA']) & (data['close'] < data['Long_MA']))
    data['Position_s2'] = 0
    data.loc[buy_signal, 'Position_s2'] = 1
    data.loc[sell_signal, 'Position_s2'] = -1
    data['Buy_Signal_s2'] = buy_signal
    data['Sell_Signal_s2'] = sell_signal
    return data

def generate_signals_macd(data):
    cross_above = (data['DIF'].shift(1) < data['DEA'].shift(1)) & (data['DIF'] > data['DEA'])
    cross_below = (data['DIF'].shift(1) > data['DEA'].shift(1)) & (data['DIF'] < data['DEA'])
    macd_cross_above = (data['MACD'].shift(1) < 0) & (data['MACD'] > 0)
    macd_cross_below = (data['MACD'].shift(1) > 0) & (data['MACD'] < 0)
    data['Position_s3'] = 0
    data.loc[cross_above & macd_cross_above, 'Position_s3'] = 1
    data.loc[cross_below | macd_cross_below, 'Position_s3'] = -1
    return data

def generate_signals_kdj(data, period=9, k_period=3, d_period=3):
    condition_buy_set1 = (data['D'] < 80) & (data['J'] < 0)
    condition_sell_set1 = (data['D'] > 80) & (data['J'] > 100)
    condition_buy_set2 = (data['J'].shift(1) < data['D'].shift(1)) & (data['J'] > data['D']) & \
                         (data['J'].shift(1) < 50) & (data['J'] < 50) & \
                         (data['D'].shift(1) < 50) & (data['D'] < 50)
    condition_sell_set2 = (data['J'].shift(1) > data['D'].shift(1)) & (data['J'] < data['D']) & \
                          (data['J'].shift(1) > 50) & (data['J'] > 50) & \
                          (data['D'].shift(1) > 50) & (data['D'] > 50)
    data['Signal'] = 0
    data.loc[condition_buy_set1 | condition_buy_set2, 'Position_s4'] = 1
    data.loc[condition_sell_set1 | condition_sell_set2, 'Position_s4'] = -1
    return data

def plot_ma_strategy(btc_data, short_window, long_window):
    plt.figure(figsize=(14, 7))
    plt.plot(btc_data['close'], label='Bitcoin Price', color='black')
    plt.plot(btc_data['Short_MA'], label=f'{short_window}-Day MA', color='blue')
    plt.plot(btc_data['Long_MA'], label=f'{long_window}-Day MA', color='magenta')
    plt.plot(btc_data[btc_data['Position_s1'] == 1].index, btc_data['Short_MA'][btc_data['Position_s1'] == 1], '^', markersize=10, color='green', lw=0, label='Buy Signal')
    plt.plot(btc_data[btc_data['Position_s1'] == -1].index, btc_data['Short_MA'][btc_data['Position_s1'] == -1], 'v', markersize=10, color='red', lw=0, label='Sell Signal')
    plt.title('MA Strategy: Buy/Sell Signals')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.savefig("MA Strategy.png")
    plt.show()


def plot_mixed_strategy(btc_data, short_window, long_window):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.05})

    ax1.plot(btc_data['close'], label='Bitcoin Price', color='black')
    ax1.plot(btc_data['Short_MA'], label=f'{short_window}-Day MA', color='blue')
    ax1.plot(btc_data['Long_MA'], label=f'{long_window}-Day MA', color='magenta')
    ax1.plot(btc_data[btc_data['Position_s2'] == 1].index, btc_data['close'][btc_data['Position_s2'] == 1], '^', markersize=10, color='orange', label='Buy Signal')
    ax1.plot(btc_data[btc_data['Position_s2'] == -1].index, btc_data['close'][btc_data['Position_s2'] == -1], 'v', markersize=10, color='purple', label='Sell Signal')
    ax1.set_title('Mixed Strategy Signals with Technical Indicators')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')

    ax2.plot(btc_data.index, btc_data['DIF'], label='DIF', color='green')
    ax2.plot(btc_data.index, btc_data['DEA'], label='DEA', color='red')
    macd_colors = ['green' if val >= 0 else 'red' for val in btc_data['MACD']]
    ax2.bar(btc_data.index, btc_data['MACD'], color=macd_colors, label='MACD', width=0.7)
    ax2.legend(loc='upper left')
    ax2.set_ylabel('MACD')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax3.plot(btc_data['RSI'], label='RSI', color='blue')
    ax3.axhline(70, color='red', linestyle='--', linewidth=0.5)
    ax3.axhline(30, color='green', linestyle='--', linewidth=0.5)
    ax3.fill_between(btc_data.index, btc_data['RSI'], 70, where=(btc_data['RSI'] >= 70), color='red', alpha=0.5)
    ax3.fill_between(btc_data.index, btc_data['RSI'], 30, where=(btc_data['RSI'] <= 30), color='green', alpha=0.5)
    ax3.set_ylabel('RSI')
    ax3.legend(loc='upper left')

    plt.xlabel('Date')
    plt.savefig("Mixed_Strategy.png")
    plt.show()

def plot_kdj_strategy(btc_data):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    axs[0].plot(btc_data['close'], label='Bitcoin Price', color='black')
    axs[0].plot(btc_data[btc_data['Position_s4'] == 1].index, btc_data['close'][btc_data['Position_s4'] == 1], '^', markersize=10, color='cyan', lw=0, label='Buy Signal')
    axs[0].plot(btc_data[btc_data['Position_s4'] == -1].index, btc_data['close'][btc_data['Position_s4'] == -1], 'v', markersize=10, color='gray', lw=0, label='Sell Signal')
    axs[0].legend()
    axs[0].set_ylabel('Price')
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].set_title('KDJ Strategy Signals and KDJ lines')

    axs[1].plot(btc_data['K'], label='K Line', color='blue')
    axs[1].plot(btc_data['D'], label='D Line', color='magenta')
    axs[1].plot(btc_data['J'], label='J Line', color='red')
    axs[1].axhline(20, color='gray', linestyle='--', linewidth=1)
    axs[1].axhline(80, color='gray', linestyle='--', linewidth=1)
    axs[1].legend()
    axs[1].set_ylabel('KDJ')

    axs[1].set_xlabel('Date')
    plt.savefig("KDJ_Strategy.png")
    plt.show()

def plot_macd_strategy(btc_data):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    axs[0].plot(btc_data.index, btc_data['close'], label='Bitcoin Price', color='black')
    axs[0].plot(btc_data[btc_data['Position_s3'] == 1].index, btc_data['close'][btc_data['Position_s3'] == 1], '^', markersize=10, color='gold', lw=0, label='Buy Signal')
    axs[0].plot(btc_data[btc_data['Position_s3'] == -1].index, btc_data['close'][btc_data['Position_s3'] == -1], 'v', markersize=10, color='darkblue', lw=0, label='Sell Signal')
    axs[0].legend()
    axs[0].set_ylabel('Price')
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].set_title('MACD Strategy Signals and MACD lines')

    axs[1].plot(btc_data.index, btc_data['DIF'], label='DIF', color='green')
    axs[1].plot(btc_data.index, btc_data['DEA'], label='DEA', color='red')
    macd_colors = ['green' if val >= 0 else 'red' for val in btc_data['MACD']]
    axs[1].bar(btc_data.index, btc_data['MACD'], color=macd_colors, label='MACD', width=0.7)
    axs[1].legend()
    axs[1].set_ylabel('MACD')

    axs[1].set_xlabel('Date')
    plt.savefig("MACD_Strategy.png")
    plt.show()


def load_and_backtest_model():
    df = pd.read_csv("BTC_TECH.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=True)
    df = df.iloc[3200:]

    backtest_env_ppo = StockEnv(df, init_money=100000, window_size=10)
    backtest_env_dqn = StockEnv(df, init_money=100000, window_size=10)
    obs1 = backtest_env_ppo.reset()
    obs2 = backtest_env_dqn.reset()

    model_PPO = PPO.load("PPO")
    model_DQN = DQN.load("DQN")

    done = False
    while not done:
        action, _states = model_PPO.predict(obs1, deterministic=True)
        obs1, rewards, done, info = backtest_env_ppo.step(action)

    backtest_env_ppo.draw('ppo_validation_trade_backtest.png', 'ppo_validation_profit_backtest.png')

    done = False
    while not done:
        action, _states = model_DQN.predict(obs2, deterministic=True)  # 使用确定性策略进行预测
        obs2, rewards, done, info = backtest_env_dqn.step(action)

    backtest_env_dqn.draw('dqn_validation_trade_backtest.png', 'dqn_validation_profit_backtest.png')

    return backtest_env_ppo, backtest_env_dqn


if __name__ == "__main__":
    btc_data = pd.read_csv('BTC_TECH.csv')
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data.set_index('date', inplace=True)
    btc_data = btc_data.iloc[3200+5:-2]

    short_window = 5
    long_window = 10

    btc_data = generate_signals_s1(btc_data)
    btc_data = generate_signals_s2(btc_data)
    btc_data = generate_signals_macd(btc_data)
    btc_data = generate_signals_kdj(btc_data)

    initial_capital = 100000
    transaction_rate = 0.001

    portfolio = pd.DataFrame(index=btc_data.index, columns=['Position_s1', 'Position_s2', 'Position_s3', 'Position_s4', 'Holdings_s1', 'Cash_s1', 'Total_s1', 'Returns_s1', 'Holdings_s2', 'Cash_s2', 'Total_s2', 'Returns_s2',
                                                            'Holdings_s3', 'Cash_s3', 'Total_s3', 'Returns_s3', 'Holdings_s4', 'Cash_s4', 'Total_s4', 'Returns_s4'])
    portfolio['Position_s1'] = btc_data['Position_s1']
    portfolio['Position_s2'] = btc_data['Position_s2']
    portfolio['Position_s3'] = btc_data['Position_s3']
    portfolio['Position_s4'] = btc_data['Position_s4']
    portfolio[['Cash_s1', 'Cash_s2', 'Cash_s3', 'Cash_s4','Holdings_s1', 'Holdings_s2', 'Holdings_s3','Holdings_s4','Returns_s1', 'Returns_s2', 'Returns_s3', 'Returns_s4',
               'Total_s1', 'Total_s2', 'Total_s3', 'Total_s4']] = 0
    portfolio.loc[portfolio.index[0], ['Cash_s1', 'Cash_s2', 'Cash_s3', 'Cash_s4']] = initial_capital
    portfolio.loc[portfolio.index[0], ['Total_s1', 'Total_s2', 'Total_s3', 'Total_s4']] = initial_capital

    for i in range(1, len(portfolio)):
        for strategy in ['s1', 's2', 's3', 's4']:
            position_col = f'Position_{strategy}'
            holdings_col = f'Holdings_{strategy}'
            cash_col = f'Cash_{strategy}'
            total_col = f'Total_{strategy}'
            if portfolio.iloc[i][position_col] == 1 and portfolio[holdings_col][i-1] == 0:
                portfolio[holdings_col][i] = portfolio[cash_col][i-1] // btc_data.iloc[i]['close']
                portfolio[cash_col][i] = portfolio[cash_col][i-1] - (portfolio[holdings_col][i] * btc_data['close'][i] * (1 + transaction_rate))
            elif portfolio.iloc[i][position_col] == -1 and portfolio[holdings_col][i-1] != 0:
                portfolio[cash_col][i] = portfolio[cash_col][i-1] + (portfolio[holdings_col][i-1] * btc_data['close'][i] * (1 - transaction_rate))
                portfolio[holdings_col][i] = 0
            else:
                portfolio[holdings_col][i] = portfolio[holdings_col][i-1]
                portfolio[cash_col][i] = portfolio[cash_col][i-1]
            portfolio[total_col][i] = portfolio[holdings_col][i] * btc_data['close'][i] + portfolio[cash_col][i]

    portfolio['Returns_s1'] = portfolio['Total_s1'].pct_change()
    portfolio['Returns_s2'] = portfolio['Total_s2'].pct_change()
    portfolio['Returns_s3'] = portfolio['Total_s3'].pct_change()
    portfolio['Returns_s4'] = portfolio['Total_s3'].pct_change()

    # Plotting MA strategy
    plot_ma_strategy(btc_data, short_window, long_window)

    # Plotting MACD Strategy
    plot_macd_strategy(btc_data)

    # Plotting KDJ Strategy
    plot_kdj_strategy(btc_data)

    # Plotting Mixed Strategy
    plot_mixed_strategy(btc_data, short_window, long_window)

    ppo_env, dqn_env = load_and_backtest_model()

    sell_point_ppo, buy_point_ppo, profit_rate_ppo, profit_rate_stock_ppo = ppo_env.get_info()
    sell_point_dqn, buy_point_dqn, profit_rate_dqn, profit_rate_stock_dqn = dqn_env.get_info()
    profit_rate_series_ppo = pd.Series(profit_rate_ppo, index=portfolio.index)
    profit_rate_series_dqn = pd.Series(profit_rate_dqn, index=portfolio.index)

    plt.figure(figsize=(13, 7))
    # 策略1的账户价值变化
    plt.plot((portfolio['Total_s1'] - initial_capital)/ initial_capital, label='MA Strategy', color='green')
    # 策略2的账户价值变化
    plt.plot((portfolio['Total_s2'] - initial_capital) / initial_capital, label='MACD & RSI & MA Strategy', color='orange')
    # 策略3的账户价值变化
    plt.plot((portfolio['Total_s3'] - initial_capital) / initial_capital, label='MACD Strategy', color='blue')
    # 策略4的账户价值变化
    plt.plot((portfolio['Total_s4'] - initial_capital) / initial_capital, label='KDJ Strategy', color='navy')
    # the model performance
    plt.plot(profit_rate_series_dqn, label='DQN Agent', color='gold')
    # the model performance
    plt.plot(profit_rate_series_ppo, label='PPO Agent', color='lime')
    # 比特币大盘价值变化
    plt.plot((btc_data['close'] - btc_data['close'].iloc[0]) / btc_data['close'].iloc[0], label='Bitcoin Market Price', color='red')

    plt.title('Portfolio Value vs. Market Value: Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Profit_rate')
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    plt.legend()
    plt.savefig("result.png")
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
from matplotlib import ticker
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, init_money=10000, window_size=6):
        super(StockEnv, self).__init__()

        self.action_space = spaces.Discrete(3)  # 0: 保持，1: 买入，2: 卖出
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 11,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 16), dtype=np.float32)

        self.trend = df['close'].values # 收盘数据
        self.volumn = df['volume'].values / 100000
        self.high = df['high'].values
        self.low = df['low'].values
        self.open = df['open'].values
        self.rsi = df['RSI'].values
        self.ma30 = df['MA_30'].values
        self.ma90 = df['MA_90'].values
        self.macd = df['MACD'].values
        self.k = df['K'].values
        self.d = df['D'].values
        self.j = df['J'].values


        self.no_trade_days = 0
        self.value_change = 0

        self.df = df #数据的DataFrame
        self.init_money = init_money # 初始化资金

        self.window_size = window_size #滑动窗口大小
        self.half_window = window_size // 2

        self.buy_rate = 0.001  # 买入费率
        self.buy_min = 0  # 最小买入费率
        self.sell_rate = 0.001  # 卖出费率
        self.sell_min = 0  # 最小卖出费率
        self.stamp_duty = 0  # 印花税
        # self.short_rate = 0.0004

    def reset(self):
        self.hold_money = self.init_money # 持有资金
        self.buy_num = 0 # 买入数量
        self.hold_num = 0 # 持有股票数量
        self.stock_value = 0 # 持有股票总市值
        self.market_value = 0 # 总市值（加上现金）
        self.last_value = self.init_money # 上一天市值
        self.total_profit = 0 # 总盈利
        self.t = self.window_size // 2 # 时间
        self.reward = 0 # 收益
        self.no_trade_days = 0
        self.value_change = 0
        # self.inventory = []

        self.states_sell = [] #卖股票时间
        self.states_buy = [] #买股票时间

        # self.states_open_short = []
        # self.states_close_short = []
        # self.short_inventory = []

        self.profit_rate_account = [] # 账号盈利
        self.profit_rate_stock = [] # 股票波动情况
        return self.get_state(self.t)


    def get_state(self, t):
        window_size = self.window_size
        d = t - window_size + 1

        # 初始化状态数组，考虑到所有特征
        res = np.zeros((1, window_size, 16))

        # 遍历窗口中的每个时间步
        for i in range(window_size):
            idx = max(d + i, 0)  # 确保索引非负
            # 对于窗口中的每个时间步，计算其特征
            # 使用动态索引idx而不是静态的t，以获取正确的历史值
            price_diff = (self.trend[idx + 1] - self.trend[idx]) / (self.trend[idx] + 0.00001) if idx + 1 < len(self.trend) else 0
            # price_diff = (self.trend[idx] - self.trend[idx - 1]) / (self.trend[idx - 1]) if idx - 1 >= 0 else 0
            res[0, i, :] = [
                price_diff,
                self.total_profit / self.init_money / 10,
                self.value_change / self.init_money / 10,
                self.volumn[idx] / 1000000,
                self.trend[idx] / 100000,
                self.open[idx] / 100000,
                self.high[idx] / 100000,
                self.low[idx] / 100000,
                self.ma30[idx] / 100000,
                self.ma90[idx] / 100000,
                self.macd[idx] / 10000,
                self.rsi[idx] / 100,
                self.no_trade_days,
                self.k[idx] / 100,
                self.d[idx] / 100,
                self.j[idx] / 100
            ]

        return res


    def buy_stock(self):
        # 买入股票

        self.buy_num = self.hold_money / self.trend[self.t] // 0.01
        self.buy_num = self.buy_num * 0.01

        # 计算手续费等
        tmp_money = self.trend[self.t] * self.buy_num
        service_change = tmp_money * self.buy_rate
        if service_change < self.buy_min:
            service_change = self.buy_min

        # 就少买1手
        if service_change + tmp_money > self.hold_money:
            # self.buy_num = self.buy_num - 100
            self.buy_num = self.buy_num - 0.01
        tmp_money = self.trend[self.t] * self.buy_num
        service_change = tmp_money * self.buy_rate
        if service_change < self.buy_min:
            service_change = self.buy_min

        self.hold_num += self.buy_num
        self.stock_value += self.trend[self.t] * self.buy_num
        self.hold_money = self.hold_money - self.trend[self.t] * \
            self.buy_num - service_change
        self.states_buy.append(self.t)

    def sell_stock(self, sell_num):
        tmp_money = sell_num * self.trend[self.t]
        service_change = tmp_money * self.sell_rate
        if service_change < self.sell_min:
            service_change = self.sell_min
        stamp_duty = self.stamp_duty * tmp_money
        self.hold_money = self.hold_money + tmp_money - service_change - stamp_duty
        self.hold_num = 0
        self.stock_value = 0
        self.states_sell.append(self.t)


    # def open_short(self):
    #     # 计算可以做空的股数，这里以账户余额为100%的保证金计算
    #     num_shares = self.hold_money / self.trend[self.t] // 0.01
    #     num_shares = self.buy_num * 0.01

    #     # 记录这次做空操作的头寸信息
    #     self.short_inventory.append({'num_shares': num_shares, 'avg_open_price': self.trend[self.t], 'open_index': self.t})
    #     self.states_open_short.append(self.t)

    # def close_short(self):
    #     if not self.short_inventory:
    #         return  # 如果没有空头头寸，直接返回

    #     fee = 0

    #     for short_position in self.short_inventory:
    #         fee = short_position['num_shares'] * short_position['avg_open_price'] * self.short_rate
    #         # 计算盈亏
    #         profit = (short_position['avg_open_price'] - self.trend[self.t]) * short_position['num_shares'] - fee
    #         # 更新账户余额
    #         self.hold_money += profit
    #     # 清空空头头寸
    #     self.short_inventory = []
    #     return fee


    def trick(self):
        if self.df['close'][self.t] >= self.df['ma21'][self.t]:
            return True
        else:
            return False

    def step(self, action, show_log=False, my_trick=False):

        service_fee = 0

        if action == 1 and self.hold_money >= (self.trend[self.t]*0.01 + \
            max(self.buy_min, self.trend[self.t]*0.01*self.buy_rate)) and self.t < (len(self.trend) - self.half_window):
            buy_ = True
            if my_trick and not self.trick():
                # 如果使用自己的触发器并不能出发买入条件，就不买
                buy_ = False
            if buy_ :
                self.buy_stock()
                self.no_trade_days = 0
                service_fee = self.buy_num * self.trend[self.t] * self.buy_rate
                if show_log:
                    print('day:%d, buy price:%f, buy num:%f, hold num:%f, hold money:%.3f'% \
                          (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))

        elif action == 2 and self.hold_num > 0:
            # 卖出股票
            holdings = self.hold_num
            self.sell_stock(self.hold_num)
            self.no_trade_days = 0
            service_fee = holdings * self.trend[self.t] * self.sell_rate

            # # sell strategy
            # sell_num = self.hold_num / 5 * 4
            # self.sell_stock(sell_num)
            if show_log:
                print(
                    'day:%d, sell price:%f, total balance %f,'
                    % (self.t, self.trend[self.t], self.hold_money)
                )

        else:
            self.no_trade_days += 1
            if my_trick and self.hold_num>0 and not self.trick():
                self.sell_stock(self.hold_num)
                if show_log:
                    print(
                        'day:%d, sell price:%f, total balance %f,'
                        % (self.t, self.trend[self.t], self.hold_money)
                    )


        # elif action == 3 and self.hold_num == 0 and not self.short_inventory:
        #     self.open_short()
        #     self.no_trade_days = 0

        # elif action == 4 and self.short_inventory:
        #     service_fee = self.close_short()
        #     self.no_trade_days = 0




        self.stock_value = self.trend[self.t] * self.hold_num
        self.market_value = self.stock_value + self.hold_money
        self.total_profit = self.market_value - self.init_money


        reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
        if np.abs(reward)<=0.015:
            reward = reward * 0.2
        elif np.abs(reward)<=0.03:
            reward = reward * 0.7
        elif np.abs(reward)>=0.05:
            if reward < 0 :
                reward = (reward+0.05) * 0.1 - 0.05
            else:
                reward = (reward-0.05) * 0.1 + 0.05

        # reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
        # if self.hold_num > 0 or action == 2:
        #     self.reward = reward
        #     if action == 2:
        #         self.reward = -self.reward
        # else:
        #     self.reward = -self.reward * 0.1
            # self.reward = 0

        self.reward = reward

        self.value_change = self.market_value - self.last_value - service_fee
        self.reward += self.value_change * 0.00001
        self.reward += (self.total_profit / self.init_money) * 0.01
        # if self.short_inventory:
        #     self.reward -= 0.1

        fall = self.trend[min(self.t + self.half_window, len(self.trend) - 1)] > self.trend[max(self.t - self.half_window, 0)]
        if self.no_trade_days > 10 and self.no_trade_days <= 30 and not fall:
            self.reward -= 1  # 减少奖励
        elif self.no_trade_days > 30 and not fall:
            self.reward -= 5

        self.last_value = self.market_value

        self.profit_rate_account.append((self.market_value - self.init_money) / self.init_money)
        self.profit_rate_stock.append((self.trend[self.t] - self.trend[0]) / self.trend[0])
        done = False
        self.t = self.t + 1
        if self.t == len(self.trend) - 2:
            done = True
        s_ = self.get_state(self.t)
        reward = self.reward

        info = {}
        info['step_reward'] = reward

        return s_, reward, done, info

    def get_info(self):
        return self.states_sell, self.states_buy, self.profit_rate_account, self.profit_rate_stock

    def draw(self, save_name1, save_name2):
        # 绘制结果
        states_sell, states_buy, profit_rate_account, profit_rate_stock = self.get_info()
        invest = profit_rate_account[-1] * 100
        total_gains = self.total_profit
        close = self.trend

        fig = plt.figure(figsize = (15,5))
        plt.plot(self.df['date'], close, color='r', lw=2.)
        plt.plot(self.df['date'], close, 'v', markersize=8, color='k', label = 'selling signal', markevery = states_sell)
        plt.plot(self.df['date'], close, '^', markersize=8, color='m', label = 'buying signal', markevery = states_buy)
        # plt.plot(close, 'o', markersize=8, color='b', label='open short signal', markevery=self.states_open_short)  # 开空信号
        # plt.plot(close, 'x', markersize=8, color='y', label='close short signal', markevery=self.states_close_short)  # 平空信号

        plt.title('total gains %f, total profit rate %f%%'%(total_gains, invest))
        plt.xlabel('days')
        plt.ylabel('profits')
        plt.legend()
        plt.xticks(rotation=45)

        plt.savefig(save_name1)

        fig = plt.figure(figsize = (15,5))
        plt.plot(self.df['date'].iloc[self.half_window:-2], profit_rate_account, label='my account')
        plt.plot(self.df['date'].iloc[self.half_window:-2], profit_rate_stock, label='stock')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plt.title('total gains %f, total profit rate %f%%'%(total_gains, invest))
        plt.xlabel('days')
        plt.ylabel('profit_rate')
        plt.legend()
        plt.xticks(rotation=45)

        plt.savefig(save_name2)


class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomLSTMExtractor, self).__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(input_size=16, hidden_size=features_dim, batch_first=True)

    def forward(self, observations):
        # LSTM期望输入的形状为(batch, seq_len, features)
        lstm_out, _ = self.lstm(observations)
        # 只关心序列的最后输出
        return lstm_out[:, -1, :]


class RewardLoggingAndSaveModelCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super(RewardLoggingAndSaveModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['infos'][0].get('step_reward')
        if reward is not None:
            self.rewards.append(reward)

        # 每5000步计算并记录平均奖励
        if len(self.rewards) % 5000 == 0:
            average_reward = sum(self.rewards[-5000:]) / 5000
            self.logger.record("custom/average_reward", average_reward)

        # 按照设定频率保存模型
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path + str(self.n_calls))

        return True

        
def train_model():
    file_path = 'BTC_TECH.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=True)
    df = df.iloc[1000:].reset_index(drop=True)

    env = StockEnv(df.iloc[:2200], init_money=100000, window_size=10)
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = {
        "features_extractor_class": CustomLSTMExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [],
    }

    model_PPO = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        n_steps=256,
        batch_size=16,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.4,
        ent_coef=0.00015,
        vf_coef=0.30244638,
        max_grad_norm=0.7,
        n_epochs=20,
        target_kl=None,
        verbose=0,
        tensorboard_log="./tensorboard_log"
    )

    model_DQN = DQN(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        buffer_size=4000,
        learning_starts=1000,
        batch_size=16,
        tau=1.0 / 10,  # Period of Q target network updates, tau corresponds to (1 - tau) * old + tau * new
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        exploration_initial_eps=1,
        exploration_final_eps=0.12,
        exploration_fraction=0.1,
        target_update_interval=10,
        max_grad_norm=10,
        tensorboard_log="./tensorboard_log",
        verbose=0
    )

    callback = RewardLoggingAndSaveModelCallback(save_path="./PPO", save_freq=100000)
    model_PPO.learn(total_timesteps=10000000, callback=callback)

    callback = RewardLoggingAndSaveModelCallback(save_path="./DQN", save_freq=100000)
    model_DQN.learn(total_timesteps=10000000, callback=callback)

if __name__ == "__main__":
    train_model()

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from main import stock
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# shares normalization factor
# 100 shares per trade
STOCKS_PERTRADE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4 #It speeds up the results


class Stock_Environment_Train(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataframe, row=0):
        # Preprocessing of Dataframe
        self.row = row
        self.dataframe = dataframe
        self.row = self.dataframe.Row[0]
        # -1=Sell, 0=Remain, 1= Buy
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        # Shape = 181: [Current Balance 0]+[ Close-Open-High-Low-Volume 1-5]+[owned shares 6]+[Histogram_Retracement 7]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        # load data from a pandas dataframe
        self.data = self.dataframe.loc[self.dataframe.index[self.row], :]
        self.terminal = False
        # initalize state
        self.state = np.array([INITIAL_ACCOUNT_BALANCE,
                     self.data.Close,
                     self.data.Open,
                     self.data.High,
                     self.data.Low,
                     self.data.Volume,
                     0,
                     self.data.Hist_Retrace], dtype=float)
#Balance: 0, Close:1, Open:2, High:3, Low: 4, Volume:5, # of actions held:6
        #Hist_retrace: 7
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self._seed()

    def _sell_stock(self, action):
        # perform sell action based on the sign of the action
        if self.state[6] > 0:
            # update balance, 1st sum price of action to balance, 2nd rest number of actions
            self.state[0] += self.state[1] * min(abs(action), self.state[6]) * (1 - TRANSACTION_FEE_PERCENT)

            self.state[6] -= min(abs(action), self.state[6])
            self.cost += self.state[1] * min(abs(action), self.state[6]) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy_stock(self, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[1] #balance/price of stock
        # update balance
        self.state[0] -= self.state[1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

        self.state[6] += min(available_amount, action)

        self.cost += self.state[1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        self.terminal = self.row >= len(self.dataframe.Row)-2

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0] + self.state[1]*self.state[6]
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * STOCKS_PERTRADE
            begin_total_asset = self.state[0] + (self.state[1]) * (self.state[6]) #We begin the trade operation with this total money
            # print("begin_total_asset:{}".format(begin_total_asset))
            self._sell_stock(actions) #if action is + it will buy
            self._buy_stock(actions) #if action is - it will sell, if 0 won't do anything
            self.row += 1
            self.data = self.dataframe.loc[self.dataframe.index[self.row], :] #select the next row of observations
            #next observation
            self.state = np.array([self.state[0],
                         self.data.Close,
                         self.data.Open,
                         self.data.High,
                         self.data.Low,
                         self.data.Volume,
                         self.state[6],
                         self.data.Hist_Retrace], dtype=float)
            end_total_asset = self.state[0] + (self.state[1]) * (self.state[6])
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.row = self.dataframe.Row[0]
        self.data = self.dataframe.loc[self.dataframe.index[self.row], :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = np.array([INITIAL_ACCOUNT_BALANCE,
                     self.data.Close,
                     self.data.Open,
                     self.data.High,
                     self.data.Low,
                     self.data.Volume,
                     0,
                     self.data.Hist_Retrace], dtype=float)
        # iteration += 1
        self.state = np.array(self.state)
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
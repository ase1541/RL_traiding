import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from main import stock

"""MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000"""


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataframe, features):
        super(StockTradingEnv, self).__init__()
        self.dataframe = dataframe #.drop(columns=['','',''])
        self.dataframe['Position'] = 0 #Add a column that shows the positions taken
        self.features = features
        # Actions of the format Buy, Sell, Out
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=int)
        #Values for the last five prices (Stock, Open, High, Low, Volume, Histogram_retracement )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6, 6), dtype=np.float16)
        self.trades=0
        self.days_per_trade=[]
        self.done = False
        self.days_in_position=0

    def _next_observation(self):
        # Get the stock data points for the last 5 days
        obs = np.array([
            self.dataframe.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step ], stock].values,
            self.features.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step ], 'Open'].values,
            self.features.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step ], 'High'].values,
            self.features.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step ], 'Low'].values,
            self.features.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step ], 'Volume'].values,
            self.dataframe.loc[self.dataframe.index[self.current_step- 5]: self.dataframe.index[self.current_step], f'Histograma_MACD_{stock}'].values,
        ])
        return obs

    def _take_action(self, action):
        #Short
        if (action == -1) & ((self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == 0)):
            self.days_in_position = 0
            self.dataframe.loc[self.dataframe.index[self.current_step],'Position'] = -1
            self.days_in_position += 1
            self.trades += 1
        #Long
        if (action == 1) & ((self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == 0)):

            self.days_in_position = 0
            self.dataframe.loc[self.dataframe.index[self.current_step],'Position'] = 1
            self.days_in_position += 1
            self.trades += 1

        if (action == 0) & ((self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == 1)
                            or (self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == -1)):
            self.days_per_trade.append(self.days_in_position)
            self.days_in_position = 0
            self.dataframe.loc[self.dataframe.index[self.current_step], 'Position'] = 0

        if (action == 0) & ((self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == 0)):
            self.days_in_position = 0
            self.dataframe.loc[self.dataframe.index[self.current_step], 'Position'] = 0

        if ((action == 1 and (self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == 1) )
                or (action == -1 and (self.dataframe.loc[self.dataframe.index[self.current_step-1],'Position'] == -1) )):
            self.days_in_position += 1
            self.dataframe.loc[self.dataframe.index[self.current_step], 'Position'] = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if self.current_step == len(self.dataframe[stock])-2:
            self.done = True
        else:
            self.done = False
        retr_short = self.dataframe[f'Daily_returns_{stock}'].loc[self.dataframe['Position']==-1]
        retr_long = self.dataframe[f'Daily_returns_{stock}'].loc[self.dataframe['Position'] == 1]
        reward = (retr_long+retr_short) - self.last_reward
        self.last_reward = reward
        obs = self._next_observation()

        return obs, reward, self.done

    def reset(self):
        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            5, len(self.dataframe[stock]))
        self.trades = 0
        self.days_per_trade = []
        self.done = False
        self.last_reward=0
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Number of trades done so far: {self.trades}')
        print(f'Days trade held last time: {self.days_per_trade}')
        print(f'Current days on position: {self.days_in_position}')


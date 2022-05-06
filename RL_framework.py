from gym.utils import seeding
from gym import spaces
import matplotlib
import pandas as pd
import numpy as np
import time
import gym
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RL models from stable-baselines
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


def train_A2C(env_train,model_name,timesteps =25000):
    """A2C model"""
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"Training/A2C_{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_DDPG(env_train,model_name,timesteps =25000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"Training/DDPG_{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train,model_name,timesteps =25000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model



# shares normalization factor
# 100 shares per trade
STOCKS_PERTRADE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 100000
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4 #It speeds up the results

class Stock_Environment_Train(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataframe, row=0, type = "train", model_name = "A2C"):
        # Preprocessing of Dataframe
        self.row = row
        self.type = type #val, train or test
        self.model_name = model_name
        self.dataframe = dataframe
        self.row = 0 #self.dataframe.Row[0]
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
        self.asset_date = [self.dataframe.index[self.row]]
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
            plt.plot(self.asset_date,self.asset_memory,'b')
            plt.savefig(f'results/account_value_{self.type}_{self.model_name}.png')
            plt.close()
            end_total_asset = self.state[0] + self.state[1]*self.state[6]
            df_total_value = pd.DataFrame({"Date": self.asset_date, "Balance": self.asset_memory})
            df_total_value.set_index("Date")
            df_total_value.to_csv(f"results/account_value_{self.type}_{self.model_name}.csv")
            df_total_value['daily_return'] = df_total_value["Balance"].pct_change(1)
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
            end_date = self.dataframe.index[self.row]
            self.asset_memory.append(end_total_asset)
            self.asset_date.append(end_date)
            self.reward = (end_total_asset - begin_total_asset) #*(end_total_asset-INITIAL_ACCOUNT_BALANCE)
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.row = 0 #self.dataframe.Row[0]
        self.data = self.dataframe.loc[self.dataframe.index[self.row], :]
        self.asset_date = [self.dataframe.index[self.row]]
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

## Some functions
def validation(model, model_name, env_val):
    ###validation process###
    test_obs = env_val.reset()
    terminal = False
    while not terminal:
        action, _states = model.predict(test_obs)
        test_obs, rewards, terminal, info = env_val.step(action)
    sharpe = get_validation_sharpe(model_name)

    return sharpe


def get_validation_sharpe(model_name):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv(f"results/account_value_val_{model_name}.csv")
    df_total_value.columns = ['Trade','Date','Balance']
    df_total_value.set_index = ('Date')
    df_total_value['daily_return'] = df_total_value['Balance'].pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe

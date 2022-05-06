#If you want to check the env
"""from stock_env import Stock_Environment_Train
from stable_baselines3.common.env_checker import check_env
from main import dataframe
env = Stock_Environment_Train(dataframe)
check_env(env, warn=True)"""

#If you want to test env
"""import random
from main import dataframe
from stock_env import Stock_Environment_Train
episodes = 25
env = Stock_Environment_Train(dataframe)
for episode in range(1, episodes+1):
    obs = env.reset()
    terminal = False
    score = 0

    while not terminal:
        env.render() #we get observations for our obs space
        action= random.choice([-1,0,1])
        obs, reward, terminal, _ = env.step(action)
        score += reward
    print('Score: {} Episode:{} / {}'.format(score,episode,episodes))"""

from main import dataframe
from RL_framework import Stock_Environment_Train, validation, train_DDPG, train_PPO, train_A2C
env = Stock_Environment_Train(dataframe)


#Training data:(2007-12-06 ->2017-01-01)
train_data = dataframe.loc[(dataframe.index >= "2007-12-06") & (dataframe.index < "2017-12-31")]
#validation data:(2017-12-31 ->2018-01-01)
val_data = dataframe.loc[(dataframe.index > "2017-12-31") & (dataframe.index <= "2018-12-31")]
# test data: (2018-01-01 -> now)
test_data = dataframe.loc[(dataframe.index > "2017-12-31")]

##TRAINING
env_train_PPO = Stock_Environment_Train(train_data, type="train", model_name="PPO")
env_train_A2C = Stock_Environment_Train(train_data, type="train", model_name="A2C")
env_train_DDPG = Stock_Environment_Train(train_data, type="train", model_name="DDPG")

env_val_PPO = Stock_Environment_Train(val_data, type="val", model_name="PPO")
env_val_A2C = Stock_Environment_Train(val_data, type="val", model_name="A2C")

env_test = Stock_Environment_Train(test_data, type="test", model_name="PPO")

print("======Model training=======")
model_PPO = train_PPO(env_train_PPO,"PPO",timesteps =2000)
model_PPO.save(f"Training/PPO")
model_A2C = train_A2C(env_train_A2C,"A2C",timesteps =2000)
model_A2C.save(f"Training/A2C")
#model_DDPG = train_DDPG(env_train_DDPG,"DDPG",timesteps =25000)
#model_DDPG.save(f"Training/DDPG")

##VALIDATION
print("======Model Validation=======")
sharpe_PPO = validation(model_PPO,"PPO",env_val_PPO)
sharpe_A2C = validation(model_A2C,"A2C",env_val_A2C)
#sharpe_DDPG = validation(model_DDPG,"DDPG",val_data,env_val)

print(f"Sharpes: PPO={sharpe_PPO} A2C={sharpe_A2C} DDPG={0}")#sharpe_DDPG}")"""
#If we want to evaluate performance of the agent

##TEST
episodes = 5
for episode in range(1, episodes+1):
    obs = env_test.reset()
    terminal = False
    score = 0

    while not terminal:
        env.render() #we get observations for our obs space
        action, _= model_PPO.predict(obs)
        obs, reward, terminal, _ = env_test.step(action)
        score += reward
    print('Score: {} Episode:{} / {}'.format(score,episode,episodes))
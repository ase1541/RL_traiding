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



import random
from main import dataframe
from stock_env import Stock_Environment_Train
from stable_baselines3 import PPO
episodes = 25
env = Stock_Environment_Train(dataframe)
model = PPO('MlpPolicy', env, verbose =1)
model.learn(total_timesteps=20000)

#If we want to evaluate performance of the agent
"""for episode in range(1, episodes+1):
    obs = env.reset()
    terminal = False
    score = 0

    while not terminal:
        env.render() #we get observations for our obs space
        action, _= model.predict(obs)
        obs, reward, terminal, _ = env.step(action)
        score += reward
    print('Score: {} Episode:{} / {}'.format(score,episode,episodes))"""
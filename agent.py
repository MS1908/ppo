def set_seed(rand_seed):
    set_global_seeds(100)
    env.env_method('seed', rand_seed)
    np.random.seed(rand_seed)
    os.environ['PYTHONHASHSEED']=str(rand_seed)
    model.set_random_seed(rand_seed)
import gym
import gym_offload_autoscale
import numpy as np
import pandas as pd
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

env = gym.make('offload-autoscale-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])
rand_seed = 1234
model = PPO2(MlpPolicy, env, verbose=1, seed=rand_seed)
model.learn(total_timesteps=100000)
# print(env.env_method('fixed_action_cal', 1000))
# exit()
rewards_list_ppo = []
avg_rewards_ppo = []
rewards_list_random = []
avg_rewards_random = []
rewards_list_myopic = []
avg_rewards_myopic = []
rewards_list_fixed_1 = []
avg_rewards_fixed_1 = []
rewards_list_fixed_2 = []
avg_rewards_fixed_2 = []

s = 2
t_range = 10000

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('myopic_action_cal')
    obs, rewards, dones, info = env.step(action)
    rewards_list_myopic.append(1 / rewards/ s)
    avg_rewards_myopic.append(np.mean(rewards_list_myopic[:]))
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_1.append(1 / rewards/ s)
    avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 1000)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_2.append(1 / rewards/ s)
    avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    # action, _states = model.predict(obs, deterministic=True)
    action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_random.append(1 / rewards/ s)
    avg_rewards_random.append(np.mean(rewards_list_random[:]))
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action, _states = model.predict(obs, deterministic=True)
    # action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards/ s)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    if dones: env.reset()
    # env.render()
print('--RESULTS--')
print('{:15}{:30}'.format('method','time average cost'))
print('{:15}{:<30}'.format('myopic',avg_rewards_myopic[-1]))
print('{:15}{:<30}'.format('fixed 0.4kW',avg_rewards_fixed_1[-1]))
print('{:15}{:<30}'.format('fixed 1kW',avg_rewards_fixed_2[-1]))
print('{:15}{:<30}'.format('random',avg_rewards_random[-1]))
print('{:15}{:<30}'.format('PPO', avg_rewards_ppo[-1]))
# print('{:>10}{:>10}{:>10}{:>10}'.format("fixed 0.4kW", "fixed 1kW", "random", "PPO"))
# print('{:>10.8} {:>10.8} {:>10.8} {:>10.8}'.format(, avg_rewards_fixed_2[-1], , avg_rewards_ppo[-1]))
import matplotlib.pyplot as plt

df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot( 'x', 'y_1', data=df, marker='o', markevery = 10, color='red', linewidth=1, label="ppo")
plt.plot( 'x', 'y_2', data=df, marker='^', markevery = 10, color='olive', linewidth=1, label="random")
plt.plot( 'x', 'y_3', data=df, marker='s', markevery = 10, color='cyan', linewidth=1, label="myopic")
plt.plot( 'x', 'y_4', data=df, marker='*', markevery = 10, color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot( 'x', 'y_5', data=df, marker='+', markevery = 10, color='navy', linewidth=1, label="fixed 1kW")
plt.legend()
plt.grid()
plt.show()
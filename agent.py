import gym
import gym_offload_autoscale
import numpy as np
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('offload-autoscale-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
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
obs = env.reset()
t = 0
t_range = 10000
# for i in range(t_range):
#     action = env.env_method('myopic_action_cal')
#     obs, rewards, dones, info = env.step(action)
#     rewards_list_myopic.append(1 / rewards)
#     avg_rewards_myopic.append(np.mean(rewards_list_myopic[-1000:-1]))
#     if dones: env.reset()

energy_rewards_ppo = []
time_rewards_ppo = []
energy_rewards_fixed1 = []
time_rewards_fixed1 = []
energy_rewards_fixed2 = []
time_rewards_fixed2 = []
energy_rewards_random = []
time_rewards_random = []
avg_energy_rewards_ppo = []
avg_time_rewards_ppo = []
avg_energy_rewards_fixed1 = []
avg_time_rewards_fixed1 = []
avg_energy_rewards_fixed2 = []
avg_time_rewards_fixed2 = []
avg_energy_rewards_random = []
avg_time_rewards_random = []

for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
    # action = [0]
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_1.append(1 / rewards)
    avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
    energy_rewards_fixed1.append(env.get_attr('energy_cost'))
    time_rewards_fixed1.append(env.get_attr('time_cost'))
    avg_energy_rewards_fixed1.append(np.mean(energy_rewards_fixed1[:]))
    avg_time_rewards_fixed1.append(np.mean(time_rewards_fixed1[:]))
    if dones: env.reset()

for i in range(t_range):
    action = env.env_method('fixed_action_cal', 1000)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_2.append(1 / rewards)
    avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
    energy_rewards_fixed2.append(env.get_attr('energy_cost'))
    time_rewards_fixed2.append(env.get_attr('time_cost'))
    avg_energy_rewards_fixed2.append(np.mean(energy_rewards_fixed2[:]))
    avg_time_rewards_fixed2.append(np.mean(time_rewards_fixed2[:]))
    if dones: env.reset()

for i in range(t_range):
    action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_random.append(1 / rewards)
    avg_rewards_random.append(np.mean(rewards_list_random[:]))
    energy_rewards_random.append(env.get_attr('energy_cost'))
    time_rewards_random.append(env.get_attr('time_cost'))
    avg_energy_rewards_random.append(np.mean(energy_rewards_random[:]))
    avg_time_rewards_random.append(np.mean(time_rewards_random[:]))
    if dones: env.reset()

for i in range(t_range):
    action, _states = model.predict(obs)
    # action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    energy_rewards_ppo.append(env.get_attr('energy_cost'))
    time_rewards_ppo.append(env.get_attr('time_cost'))
    avg_energy_rewards_ppo.append(np.mean(energy_rewards_ppo[:]))
    avg_time_rewards_ppo.append(np.mean(time_rewards_ppo[:]))
    if dones: env.reset()
    # env.render()
print('--RESULTS--')
print('{:15}{:30}'.format('method','time average cost'))
print('{:15}{:<30}'.format('fixed 0.4kW',avg_rewards_fixed_1[-1]))
print('{:15}{:<30}'.format('fixed 1kW',avg_rewards_fixed_2[-1]))
print('{:15}{:<30}'.format('random',avg_rewards_random[-1]))
print('{:15}{:<30}'.format('PPO', avg_rewards_ppo[-1]))
# print('{:>10}{:>10}{:>10}{:>10}'.format("fixed 0.4kW", "fixed 1kW", "random", "PPO"))
# print('{:>10.8} {:>10.8} {:>10.8} {:>10.8}'.format(, avg_rewards_fixed_2[-1], , avg_rewards_ppo[-1]))
import matplotlib.pyplot as plt

# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot( 'x', 'y_1', data=df, marker='o', markevery = 700, color='red', linewidth=1, label="ppo")
plt.plot( 'x', 'y_2', data=df, marker='^', markevery = 700, color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='', color='lightblue', linewidth=1, label="fixed 0")
plt.plot( 'x', 'y_4', data=df, marker='*', markevery = 700, color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot( 'x', 'y_5', data=df, marker='+', markevery = 700, color='navy', linewidth=1, label="fixed 1kW")
plt.legend()
plt.grid()
plt.show()

df=pd.DataFrame({'x': range(t_range), 'y_1': avg_energy_rewards_ppo, 'y_2': avg_energy_rewards_random, 'y_4': avg_energy_rewards_fixed1, 'y_5': avg_energy_rewards_fixed2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Energy Cost")
plt.plot( 'x', 'y_1', data=df, marker='o', markevery = 700, color='red', linewidth=1, label="ppo")
plt.plot( 'x', 'y_2', data=df, marker='^', markevery = 700, color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='', color='lightblue', linewidth=1, label="fixed 0")
plt.plot( 'x', 'y_4', data=df, marker='*', markevery = 700, color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot( 'x', 'y_5', data=df, marker='+', markevery = 700, color='navy', linewidth=1, label="fixed 1kW")
plt.legend()
plt.grid()
plt.show()

df=pd.DataFrame({'x': range(t_range), 'y_1': avg_time_rewards_ppo, 'y_2': avg_time_rewards_random, 'y_4': avg_time_rewards_fixed1, 'y_5': avg_time_rewards_fixed2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Time Cost")
plt.plot( 'x', 'y_1', data=df, marker='o', markevery = 700, color='red', linewidth=1, label="ppo")
plt.plot( 'x', 'y_2', data=df, marker='^', markevery = 700, color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='', color='lightblue', linewidth=1, label="fixed 0")
plt.plot( 'x', 'y_4', data=df, marker='*', markevery = 700, color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot( 'x', 'y_5', data=df, marker='+', markevery = 700, color='navy', linewidth=1, label="fixed 1kW")
plt.legend()
plt.grid()
plt.show()
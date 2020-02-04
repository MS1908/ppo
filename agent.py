import gym
import gym_offload_autoscale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# data = []
# y = []
# for x in range(5):
#     data.append([x, x, x])
#     y.append(3 * x)
# df = pd.DataFrame(data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df.plot.area()
# plt.title('fixed 1 kW')
# plt.plot(range(5), y)
# plt.legend()
# plt.grid()
# plt.show()
# exit()

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('offload-autoscale-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

rewards_list_ppo = []
avg_rewards_ppo = []
rewards_time_list_ppo  = []
avg_rewards_time_list_ppo = []
rewards_bak_list_ppo  = []
avg_rewards_bak_list_ppo = []
rewards_bat_list_ppo  = []
avg_rewards_bat_list_ppo = []
ppo_data = []

rewards_list_random = []
avg_rewards_random = []
rewards_time_list_random  = []
avg_rewards_time_list_random = []
rewards_bak_list_random  = []
avg_rewards_bak_list_random = []
rewards_bat_list_random  = []
avg_rewards_bat_list_random = []
random_data = []
# rewards_list_myopic = []
# avg_rewards_myopic = []
rewards_list_fixed_1 = []
avg_rewards_fixed_1 = []
rewards_time_list_fixed_1 = []
avg_rewards_time_list_fixed_1 = []
rewards_bak_list_fixed_1  = []
avg_rewards_bak_list_fixed_1 = []
rewards_bat_list_fixed_1  = []
avg_rewards_bat_list_fixed_1 = []
fixed_1_data = []

rewards_list_fixed_2 = []
avg_rewards_fixed_2 = []
rewards_time_list_fixed_2 = []
avg_rewards_time_list_fixed_2 = []
rewards_bak_list_fixed_2  = []
avg_rewards_bak_list_fixed_2 = []
rewards_bat_list_fixed_2  = []
avg_rewards_bat_list_fixed_2 = []
fixed_2_data = []

obs = env.reset()
t = 0
t_range = 10000
# for i in range(t_range):
#     action = env.env_method('myopic_action_cal')
#     obs, rewards, dones, info = env.step(action)
#     rewards_list_myopic.append(1 / rewards)
#     avg_rewards_myopic.append(np.mean(rewards_list_myopic[-1000:-1]))
#     if dones: env.reset()

for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
    # action = [0]
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_1.append(1 / rewards)
    avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_1.append(t)
    avg_rewards_time_list_fixed_1.append(np.mean(rewards_time_list_fixed_1[:]))
    rewards_bak_list_fixed_1.append(bak)
    avg_rewards_bak_list_fixed_1.append(np.mean(rewards_bak_list_fixed_1[:]))
    rewards_bat_list_fixed_1.append(bat)
    avg_rewards_bat_list_fixed_1.append(np.mean(rewards_bat_list_fixed_1[:]))
    fixed_1_data.append([avg_rewards_time_list_fixed_1[-1], avg_rewards_bak_list_fixed_1[-1], avg_rewards_bat_list_fixed_1[-1]])
    if dones: env.reset()

for i in range(t_range):
    action = env.env_method('fixed_action_cal', 1000)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_2.append(1 / rewards)
    avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_2.append(t)
    avg_rewards_time_list_fixed_2.append(np.mean(rewards_time_list_fixed_2[:]))
    rewards_bak_list_fixed_2.append(bak)
    avg_rewards_bak_list_fixed_2.append(np.mean(rewards_bak_list_fixed_2[:]))
    rewards_bat_list_fixed_2.append(bat)
    avg_rewards_bat_list_fixed_2.append(np.mean(rewards_bat_list_fixed_2[:]))
    fixed_2_data.append([avg_rewards_time_list_fixed_2[-1], avg_rewards_bak_list_fixed_2[-1], avg_rewards_bat_list_fixed_2[-1]])
    if dones: env.reset()

for i in range(t_range):
    action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_random.append(1 / rewards)
    avg_rewards_random.append(np.mean(rewards_list_random[:]))
    t, bak, bat = env.render()
    rewards_time_list_random.append(t)
    avg_rewards_time_list_random.append(np.mean(rewards_time_list_random[:]))
    rewards_bak_list_random.append(bak)
    avg_rewards_bak_list_random.append(np.mean(rewards_bak_list_random[:]))
    rewards_bat_list_random.append(bat)
    avg_rewards_bat_list_random.append(np.mean(rewards_bat_list_random[:]))
    random_data.append([avg_rewards_time_list_random[-1], avg_rewards_bak_list_random[-1], avg_rewards_bat_list_random[-1]])
    if dones: env.reset()

obs = env.reset()
for i in range(t_range):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    t, bak, bat = env.render()
    rewards_time_list_ppo.append(t)
    avg_rewards_time_list_ppo.append(np.mean(rewards_time_list_ppo[:]))
    rewards_bak_list_ppo.append(bak)
    avg_rewards_bak_list_ppo.append(np.mean(rewards_bak_list_ppo[:]))
    rewards_bat_list_ppo.append(bat)
    avg_rewards_bat_list_ppo.append(np.mean(rewards_bat_list_ppo[:]))
    ppo_data.append([avg_rewards_time_list_ppo[-1], avg_rewards_bak_list_ppo[-1], avg_rewards_bat_list_ppo[-1]])
    if dones: env.reset()
    # env.render()
# print('--RESULTS--')
# print('{:15}{:30}'.format('method','time average cost'))
# # print('{:15}{:<30}'.format('fixed 0.4kW',avg_rewards_fixed_1[-1]))
# # print('{:15}{:<30}'.format('fixed 1kW',avg_rewards_fixed_2[-1]))
# # print('{:15}{:<30}'.format('random',avg_rewards_random[-1]))
# print('{:15}{:<30}'.format('PPO', avg_rewards_ppo[-1]))
# print('{:15}{:<30}'.format('t', avg_rewards_time_list_ppo[-1]))
# print('{:15}{:<30}'.format('e', avg_rewards_energy_list_ppo[-1]))
# print('{:>10}{:>10}{:>10}{:>10}'.format("fixed 0.4kW", "fixed 1kW", "random", "PPO"))
# print('{:>10.8} {:>10.8} {:>10.8} {:>10.8}'.format(, avg_rewards_fixed_2[-1], , avg_rewards_ppo[-1]))


# plt.subplot(2,2,1)
df0 = pd.DataFrame(ppo_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df0.plot.area()
# plt.plot(range(t_range), avg_rewards_ppo)
plt.grid()
plt.ylim(0,20)
plt.title('PPO')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()

# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_time_list_ppo,'y_2': avg_rewards_time_list_ppo, 'y_3': avg_rewards_energy_list_ppo})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# plt.plot( 'x', 'y_1', data=df, marker='o', markevery = 700, color='red', linewidth=1, label="ppo")
# plt.plot( 'x', 'y_2', data=df, marker='o', markevery = 700, color='cyan', linewidth=1, label="bak")
# plt.plot( 'x', 'y_3', data=df, marker='o', markevery = 700, color='g', linewidth=1, label="energy")

# plt.subplot(2,2,2)
df1 = pd.DataFrame(random_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df1.plot.area()
# plt.plot(range(t_range), avg_rewards_ppo)
plt.grid()
plt.ylim(0,20)
plt.title('random')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()

# plt.subplot(2,2,3)
df2 = pd.DataFrame(fixed_1_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df2.plot.area()
# plt.plot(range(t_range), avg_rewards_fixed_1)
plt.grid()
plt.ylim(0,20)
plt.title('fixed 0.4 kW')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()


# plt.subplot(2,2,4)
df3 = pd.DataFrame(fixed_2_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df3.plot.area()
# plt.plot(range(t_range), avg_rewards_ppo)
plt.grid()
plt.ylim(0,20)
plt.title('fixed 1 kW')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()

# df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})

# plt.plot( 'x', 'y_2', data=df, marker='^', markevery = 700, color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='', color='lightblue', linewidth=1, label="fixed 0")
# plt.plot( 'x', 'y_4', data=df, marker='*', markevery = 700, color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.plot( 'x', 'y_5', data=df, marker='+', markevery = 700, color='navy', linewidth=1, label="fixed 1kW")

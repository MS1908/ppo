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
import matplotlib.pyplot as plt


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

rewards_list_myopic = []
avg_rewards_myopic = []
rewards_time_list_myopic  = []
avg_rewards_time_list_myopic = []
rewards_bak_list_myopic  = []
avg_rewards_bak_list_myopic = []
rewards_bat_list_myopic  = []
avg_rewards_bat_list_myopic = []
myopic_data = []

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

s = 2
t_range = 10000

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('myopic_action_cal')
    obs, rewards, dones, info = env.step(action)
    rewards_list_myopic.append(1 / rewards/ s)
    avg_rewards_myopic.append(np.mean(rewards_list_myopic[:]))
    t, bak, bat = env.render()
    rewards_time_list_myopic.append(t/s)
    avg_rewards_time_list_myopic.append(np.mean(rewards_time_list_myopic[:]))
    rewards_bak_list_myopic.append(bak/s)
    avg_rewards_bak_list_myopic.append(np.mean(rewards_bak_list_myopic[:]))
    rewards_bat_list_myopic.append(bat/s)
    avg_rewards_bat_list_myopic.append(np.mean(rewards_bat_list_myopic[:]))
    myopic_data.append([avg_rewards_time_list_myopic[-1], avg_rewards_bak_list_myopic[-1], avg_rewards_bat_list_myopic[-1]])
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_1.append(1 / rewards/ s)
    avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_1.append(t/s)
    avg_rewards_time_list_fixed_1.append(np.mean(rewards_time_list_fixed_1[:]))
    rewards_bak_list_fixed_1.append(bak/s)
    avg_rewards_bak_list_fixed_1.append(np.mean(rewards_bak_list_fixed_1[:]))
    rewards_bat_list_fixed_1.append(bat/s)
    avg_rewards_bat_list_fixed_1.append(np.mean(rewards_bat_list_fixed_1[:]))
    fixed_1_data.append([avg_rewards_time_list_fixed_1[-1], avg_rewards_bak_list_fixed_1[-1], avg_rewards_bat_list_fixed_1[-1]])
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 1000)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_2.append(1 / rewards/ s)
    avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_2.append(t/s)
    avg_rewards_time_list_fixed_2.append(np.mean(rewards_time_list_fixed_2[:]))
    rewards_bak_list_fixed_2.append(bak/s)
    avg_rewards_bak_list_fixed_2.append(np.mean(rewards_bak_list_fixed_2[:]))
    rewards_bat_list_fixed_2.append(bat/s)
    avg_rewards_bat_list_fixed_2.append(np.mean(rewards_bat_list_fixed_2[:]))
    fixed_2_data.append([avg_rewards_time_list_fixed_2[-1], avg_rewards_bak_list_fixed_2[-1], avg_rewards_bat_list_fixed_2[-1]])
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_random.append(1 / rewards/ s)
    avg_rewards_random.append(np.mean(rewards_list_random[:]))
    t, bak, bat = env.render()
    rewards_time_list_random.append(t/s)
    avg_rewards_time_list_random.append(np.mean(rewards_time_list_random[:]))
    rewards_bak_list_random.append(bak/s)
    avg_rewards_bak_list_random.append(np.mean(rewards_bak_list_random[:]))
    rewards_bat_list_random.append(bat/s)
    avg_rewards_bat_list_random.append(np.mean(rewards_bat_list_random[:]))
    random_data.append([avg_rewards_time_list_random[-1], avg_rewards_bak_list_random[-1], avg_rewards_bat_list_random[-1]])
    if dones: env.reset()

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards/ s)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    t, bak, bat = env.render()
    rewards_time_list_ppo.append(t/s)
    avg_rewards_time_list_ppo.append(np.mean(rewards_time_list_ppo[:]))
    rewards_bak_list_ppo.append(bak/s)
    avg_rewards_bak_list_ppo.append(np.mean(rewards_bak_list_ppo[:]))
    rewards_bat_list_ppo.append(bat/s)
    avg_rewards_bat_list_ppo.append(np.mean(rewards_bat_list_ppo[:]))
    ppo_data.append([avg_rewards_time_list_ppo[-1], avg_rewards_bak_list_ppo[-1], avg_rewards_bat_list_ppo[-1]])
    if dones: env.reset()
    # env.render()

# print('--RESULTS--')
# print('{:15}{:30}'.format('method','time average cost'))
# print('{:15}{:<30}'.format('myopic',avg_rewards_myopic[-1]))
# print('{:15}{:<30}'.format('fixed 0.4kW',avg_rewards_fixed_1[-1]))
# print('{:15}{:<30}'.format('fixed 1kW',avg_rewards_fixed_2[-1]))
# print('{:15}{:<30}'.format('random',avg_rewards_random[-1]))
# print('{:15}{:<30}'.format('PPO', avg_rewards_ppo[-1]))

# print('--RESULTS--')
# print('{:15}{:30}'.format('method','time average cost'))
# # print('{:15}{:<30}'.format('fixed 0.4kW',avg_rewards_fixed_1[-1]))
# # print('{:15}{:<30}'.format('fixed 1kW',avg_rewards_fixed_2[-1]))
# # print('{:15}{:<30}'.format('random',avg_rewards_random[-1]))
# print('{:15}{:<30}'.format('PPO', avg_rewards_ppo[-1]))
# print('{:15}{:<30}'.format('t', avg_rewards_time_list_ppo[-1]))
# print('{:15}{:<30}'.format('e', avg_rewards_energy_list_ppo[-1])1


# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
# # df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# plt.plot( 'x', 'y_1', data=df, marker='o', markevery = int(t_range/10), color='red', linewidth=1, label="ppo")
# plt.plot( 'x', 'y_2', data=df, marker='^', markevery = int(t_range/10), color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.plot( 'x', 'y_4', data=df, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.plot( 'x', 'y_5', data=df, marker='+', markevery = int(t_range/10), color='navy', linewidth=1, label="fixed 1kW")

# plt.subplot(2,2,1)
df1 = pd.DataFrame(ppo_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df1.plot.area()
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
df2 = pd.DataFrame(random_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df2.plot.area()
# plt.plot(range(t_range), avg_rewards_random)
plt.grid()
plt.ylim(0,20)
plt.title('random')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()

df3 = pd.DataFrame(myopic_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df3.plot.area()
# plt.plot(range(t_range), avg_rewards_myopic)
plt.grid()
plt.ylim(0,20)
plt.title('myopic')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()

# plt.subplot(2,2,3)
df4 = pd.DataFrame(fixed_1_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df4.plot.area()
# plt.plot(range(t_range), avg_rewards_fixed_1)
plt.grid()
plt.ylim(0,20)
plt.title('fixed 0.4 kW')
plt.legend()
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.show()


# plt.subplot(2,2,4)
df5 = pd.DataFrame(fixed_2_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
df5.plot.area()
# plt.plot(range(t_range), avg_rewards_fixed_2)
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

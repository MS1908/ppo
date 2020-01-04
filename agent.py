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

rewards_list_ppo = []
avg_rewards_ppo = []
rewards_list_random = []
avg_rewards_random = []
rewards_list_fixed_0 = []
avg_rewards_fixed_0 = []
rewards_list_fixed_1 = []
avg_rewards_fixed_1 = []
rewards_list_fixed_2 = []
avg_rewards_fixed_2 = []
obs = env.reset()
t = 0
for i in range(10000):
    action = [0.0]
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_0.append(1 / rewards)
    avg_rewards_fixed_0.append(np.mean(rewards_list_fixed_0[:]))
    if dones: env.reset()

# for i in range(10000):
#     action = [0.5]
#     obs, rewards, dones, info = env.step(action)
#     rewards_list_fixed_1.append(1 / rewards)
#     avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
#     if dones: env.reset()

# for i in range(10000):
#     action = [1.0]
#     obs, rewards, dones, info = env.step(action)
#     rewards_list_fixed_2.append(1 / rewards)
#     avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
#     if dones: env.reset()

# for i in range(10000):
# 	action = np.random.uniform(0, 1, 1)
# 	obs, rewards, dones, info = env.step(action)
# 	rewards_list_random.append(1 / rewards)
# 	avg_rewards_random.append(np.mean(rewards_list_random[:]))
# 	if dones: env.reset()

for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    if dones: env.reset()
    t += 1
    # env.render()
import matplotlib.pyplot as plt
df=pd.DataFrame({'x': range(10000), 'y_1': avg_rewards_ppo, 'y_3': avg_rewards_fixed_0})
 # 'y_2': avg_rewards_random, 'y_3': avg_rewards_fixed_0, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot( 'x', 'y_1', data=df, marker='', color='skyblue', linewidth=1, label="ppo")
# plt.plot( 'x', 'y_2', data=df, marker='', color='olive', linewidth=1, label="random")
plt.plot( 'x', 'y_3', data=df, marker='', color='green', linewidth=1, label="fixed 0")
# plt.plot( 'x', 'y_4', data=df, marker='', color='red', linewidth=1, label="fixed 0.5")
# plt.plot( 'x', 'y_5', data=df, marker='', color='purple', linewidth=1, label="fixed 1")
plt.legend()
plt.show()
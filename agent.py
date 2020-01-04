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
obs = env.reset()
t = 0
for i in range(10000):
	action = np.random.uniform(0, 1, 1)
	obs, rewards, dones, info = env.step(action)
	rewards_list_random.append(1 / rewards)
	avg_rewards_random.append(np.mean(rewards_list_random[:]))
	if dones: env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    if dones: env.reset()
    t += 1
    # env.render()
import matplotlib.pyplot as plt
df=pd.DataFrame({'x': range(10000), 'y1': rewards_list_ppo, 'y2': avg_rewards_ppo, 'y3': rewards_list_random, 'y4': avg_rewards_random})

plt.plot( 'x', 'y1', data=df, marker='', color='skyblue', linewidth=1, label="reward_ppo")
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=1, label="average_ppo")
plt.plot( 'x', 'y3', data=df, marker='', color='skyblue', linewidth=1, label="reward_random")
plt.plot( 'x', 'y4', data=df, marker='', color='olive', linewidth=1, label="average_random")
plt.legend()
plt.show()
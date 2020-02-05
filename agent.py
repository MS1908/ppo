import gym
import gym_offload_autoscale
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = 1.0

        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=1000000)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < 20:
            return
        batch = random.sample(self.memory, 20)
        for state, action, reward, next_state, terminal in batch:
            q_upd = reward
            if not terminal:
                q_upd = (reward + 0.95 * np.amax(self.model.predict(next_state)[0]))
            q_val = self.model.predict(state)
            q_val[0][action] = q_upd
            self.model.fit(state, q_val, verbose=0)
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)

env = gym.make('offload-autoscale-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
rand_seed = 1234
env = DummyVecEnv([lambda: env])

def set_seed(rand_seed):
    set_global_seeds(100)
    env.env_method('seed', rand_seed)
    np.random.seed(rand_seed)
    os.environ['PYTHONHASHSEED']=str(rand_seed)
    model.set_random_seed(rand_seed)

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
t_range = 2000
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
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    energy_rewards_ppo.append(env.get_attr('energy_cost'))
    time_rewards_ppo.append(env.get_attr('time_cost'))
    avg_energy_rewards_ppo.append(np.mean(energy_rewards_ppo[:]))
    avg_time_rewards_ppo.append(np.mean(time_rewards_ppo[:]))
    if dones: env.reset()
    # env.render()

dqn_reward_list = []
avg_dqn_reward_list = []
def agent():
    env = gym.make('offload-autoscale-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    solver = DQNSolver(observation_space, action_space)
    # episode = 0
    accumulated_step = 0
    while True:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        terminal = False
        step = 0
        while True:
            done = False
            action = solver.act(state)
            next_state, reward, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            step += 1
            accumulated_step += 1
            # print('\tstate: ', state)
            dqn_reward_list.append(1 / reward)
            avg_dqn_reward_list.append(np.mean(dqn_reward_list[:]))
            if step >= 96:
                done = True
            solver.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                # episode += 1
                break
            solver.replay()
        if accumulated_step == t_range:
            terminal = True
        if terminal:
            return

agent()

print(avg_rewards_fixed_1[-1], avg_rewards_fixed_2[-1], avg_rewards_random[-1], avg_rewards_ppo[-1], avg_dqn_reward_list[-1])
import matplotlib.pyplot as plt

# df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
df=pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot( 'x', 'y_1', data=df, marker='o',markevery=int(t_range/10), color='red', linewidth=1, label="ppo")
plt.plot( 'x', 'y_2', data=df, marker='^', markevery=int(t_range/10), color='olive', linewidth=1, label="random")
# plt.plot( 'x', 'y_3', data=df, marker='', color='lightblue', linewidth=1, label="fixed 0")
plt.plot( 'x', 'y_4', data=df, marker='*', markevery=int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot( 'x', 'y_5', data=df, marker='+', markevery=int(t_range/10), color='navy', linewidth=1, label="fixed 1kW")

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
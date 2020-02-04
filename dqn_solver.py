import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import gym_offload_autoscale

import matplotlib.pyplot as plt
import pandas as pd

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

reward_list = []
avg_reward_list = []

def plot():
    df = pd.DataFrame({'x': range(10000), 'y': avg_reward_list})
    plt.xlabel("Time Slot")
    plt.ylabel("Average Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=700, color='navy', linewidth=1, label="q learning")
    print('Avg cost', avg_reward_list[-1])

def agent():
    env = gym.make('offload-autoscale-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    solver = DQNSolver(observation_space, action_space)
    episode = 0
    while True:
        state = env.reset()
        episode += 1
        terminal = False
        while True:
            action = solver.act(state)
            state, reward, done, info = env.step(action)
            reward_list.append(1 / reward)
            avg_reward_list.append(np.mean(reward_list[:]))
            if done:
                break
        if episode == 10000:
            terminal = True
        if terminal:
            plot()
            exit()

agent()
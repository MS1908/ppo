import math
import gym
import numpy as np

class Solver:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.zip_space = zip(observation_space, action_space)

        self.cost_exp = {0 for (s, a) in self.zip_space}
        # the space of pds is the same with the observation space
        self.pds_value = {0 for s in self.observation_space}
        self.normal_value = {0 for s in self.observation_space}

        # Value for delta = ? and learning rate = ?
        self.delta = 0.9
        self.learning_rate = 0.9995

    def convert_to_pds(self, d_op, action):
        res = self.observation_space
        if res[1] >= d_op:
            res[1] = max(res[1] - d_op - action, 0)
        return res

    def act(self, state):
        action = int(1e9)
        for a in self.action_space:
            action = min(action, self.cost_exp + self.delta * self.pds_value)
        return action

    # Batch update
    def update(self, time):
        self.cost_exp = (1 - np.pow(self.learning_rate, time)) * self.cost_exp + np.pow(self.learning_rate, time) * 
        pass

def offload_autoscale_agent():
    env = gym.make('offload-autoscale-discrete-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    solver = Solver(observation_space, action_space)
    while True:
        state = env.reset()
        terminal = False # terminal condition of algo
        while True:
            action = solver.act(state)
            state, reward, done, info = env.step(action)
            solver.update()
            if done:

                break
        if terminal:
            exit()
        pass
    pass
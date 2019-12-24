import gym
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.optimize import minimize

class OffloadAutoscaleEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # kWh
        self.server_service_rate = 20  # units/sec
        self.server_power_consumption = 150

        self.lamda_high = 1  # units/second
        self.lamda_low = 1
        self.b_high = self.batery_capacity / self.timeslot  # W
        self.b_low = 0
        self.e_low = 0
        self.h_high = 1  # ms/unit
        self.h_low = 0
        self.e_high = 1

        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 10

        # power model
        self.d_sta = 300
        self.coef_dyn = 10
        # self.b_com = 10

        self.time_step = 0

        r_high = np.array([
            self.lamda_high,
            self.b_high,
            self.h_high,
            self.e_low])
        r_low = np.array([
            self.lamda_low,
            self.b_low,
            self.h_low,
            self.e_high])
        self.observation_space = spaces.Box(low=r_low, high=r_high)
        self.action_space = spaces.Box(low=0.0, high=self.b_high, shape=(1,), dtype=np.float32)
        self.state = [0, 0, 0, 0]

        # self.state.append(np.array(obs[1]))
        self.time = 0
        self.episode = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_e(self):
        if self.time >= 9 and self.time < 15:
            return 1
        if self.time < 6 or self.time >= 18:
            return 0
        return 1

    def get_g(self, e):
        e = int(e)
        if e == 0:
            return np.random.exponential(60)
        if e == 1:
            return np.random.normal(520, 130)
        return np.random.normal(800, 95)

    def get_lambda(self):
        return np.random.uniform(self.lamda_low, self.lamda_high)

    def get_h(self):
        return np.random.uniform(self.h_low, self.h_high)

    def get_time(self):
        self.time += 0.25
        if self.time == 24:
            self.time = 0

    def cal(self, action):
        opt_val = math.inf
        ans = [0, 0]
        ok = False
        done = False
        for number_of_server in range(1, self.max_number_of_server + 1):
            # coeff = [1, 1, 1 + self.server_power_consumption * number_of_server - action]
            # if action < 150 * number_of_server + 1:
            #     return [0, 0]
            # local_workload = np.log(action - 150 * number_of_server)
            if action < 150 * number_of_server:
                return [0, 0]
            local_workload = (action - 150 * number_of_server) / 10
            if not isinstance(local_workload, complex):
                cost_delay_local = local_workload / (number_of_server * self.server_service_rate - local_workload)
                cost_delay_cloud = (self.state[0] - local_workload) * self.state[2]
                # print('Cost delay ' + str(cost_delay_local + cost_delay_cloud))
                if opt_val > cost_delay_local + cost_delay_cloud:
                    ans = [number_of_server, local_workload]
                    opt_val = cost_delay_local + cost_delay_cloud
        if ok:
            return ans, done
        else:
            done = True
            return [0, 0], done
    
    def power_model(self, action):
        d_dyn = self.coef_dyn * self.state[0]
        d_op = d_dyn + self.d_sta
        params = [0, 0]
        params, done = self.cal(action) 
        number_of_server = params[0]
        local_workload = params[1]
        d_com = action
        return d_op, d_com, d_op + d_com, number_of_server, local_workload, done

    def get_b(self, state, action, g, d_op, d):
        b = state[1]
        if d_op > b:
            # print('unused batery')
            return b + g
        else:
            if g >= d:
                # print('recharge batery')
                return np.maximum(self.b_high, b + g - d)
            else:
                # print('discharge batery')
                return b + g - d
    # constraints for delay local function

    def reward_func(self, action, g, d_op, d, number_of_server, local_workload):
        b = self.state[1]
        act = [number_of_server, local_workload]
        if act == [0, 0]:
            cost_delay_local = 0
        else:
            cost_delay_local = local_workload / (number_of_server * self.server_service_rate - local_workload)
        cost_delay_cloud = (self.state[0] - local_workload) * self.state[2]
        cost_delay_wireless = 0
        cost_delay = cost_delay_local + cost_delay_cloud + cost_delay_wireless
        if d_op > b:
            cost_batery = 0
            cost_bak = self.back_up_cost_coef * d_op
        else:
            cost_batery = self.normalized_unit_depreciation_cost * np.maximum(d - g, 0)
            cost_bak = 0
        cost = cost_delay + cost_batery + cost_bak
        return cost

    def step(self, action):
        done = False
<<<<<<< Updated upstream
        action = float(action)
        # self.time_step += 1
=======
        action = self.get_action(float(action))
        # if math.isnan(action):
        #     done = True
        #     self.episode += 1
        #     reward = 1e9
        #     return self.state, reward, done, {}
        
        self.time_step += 1
>>>>>>> Stashed changes
        self.get_time()
        state = self.state
        print('\tstate: ',state)
        # print('\ttime: ',self.time)
        g_t = self.get_g(state[3])
        # print('\tget ', g_t)
        # print('\taction: ', action)
        d_op, d_com, d, number_of_server, local_workload = self.power_model(action)
        print('\t{0:10}{1:10}{2:10}{3:20}{4:10}'.format('d_op','d_com','d','number_server','local_workload'))
        print('\t{0:<10.3f}{1:<10.3f}{2:<10.3f}{3:<20.3f}{4:<10.3f}'.format(d_op, d_com, d, number_of_server, local_workload))
        reward = self.reward_func(action, g_t, d_op, d, number_of_server, local_workload)
        lambda_t = self.get_lambda()
        b_t = self.get_b(state, action, g_t, d_op, d)
        h_t = self.get_h()
        e_t = self.get_e()
        self.state = np.array([lambda_t, b_t, h_t, e_t])
        print('\tnew state: ', self.state)
        print('\treward: ', reward)
        if b_t < 0:
            done = True
            reward = 1e18
            self.episode += 1
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.lamda_low, self.b_low, self.h_low, self.e_low])
        self.time_step = 0
        self.time = 0

if __name__ == '__main__':
    MyEnv = OffloadAutoscaleEnv()
    MyEnv.reset()
    # # obs = MyEnv.observation_space.sample()
    # # print('debug: ', obs)
    # # print(MyEnv.observation_space.sample())
    for i in range(1000):
        action = MyEnv.action_space.sample()
        state, reward, _, _ = MyEnv.step(action)
    #     print('STEP: ', i)
    #     state, reward = MyEnv.step(i+3000)
    #     # state, reward = MyEnv.step(MyEnv.state[1])
    #     # print(state)
    #     # print(reward)

import gym
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class OffloadAutoscaleDiscreteEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # kWh
        self.server_service_rate = 20  # units/sec
        self.server_power_consumption = 150

        self.lamda_high = 100  # units/second
        self.lamda_low = 10
        self.lamda_step = 10
        self.lamda_n = round((self.lamda_high - self.lamda_low) / self.lamda_step + 1)
        self.b_high = round(self.batery_capacity / self.timeslot)  # W
        self.b_low = 0
        self.b_n = self.b_high + 1
        self.h_high = 0.06  # ms/unit
        self.h_low = 0.02
        self.h_step = 0.01
        self.h_n = round((self.h_high - self.h_low) / self.h_step + 1)
        self.e_high = 2
        self.e_low = 0
        self.e_n = self.e_high + 1

        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 10

        # power model
        self.d_sta = 300
        self.coef_dyn = 10
        # self.b_com = 10
        self.k = 15

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
        self.observation_space = spaces.MultiDiscrete([self.lamda_n, self.b_n, self.h_n, self.e_n])
        self.action_space = spaces.Discrete(self.b_n)
        self.state = [0, 0, 0, 0]

        # self.state.append(np.array(obs[1]))
        self.time = 0
        self.episode = 0
    def de_lamda(self, lamda):
        return self.lamda_low + self.lamda_step * lamda
    def get_b(self, state, g, d_op, d):
        b = state[1]
        # print('\t', end = '')
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
    def de_h(self, h):
        return self.h_low + self.h_step * h
    def de_state(self, state):
        lamda, b, h, e = state
        lamda = self.de_lamda(lamda)
        h = self.de_h(h)
        state = np.array([lamda, b, h, e])
        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_e(self):
        if self.time >= 9 and self.time < 15:
            return 2
        if self.time < 6 or self.time >= 18:
            return 0
        return 1

    def get_g(self, e):
        if e == 0:
            return np.random.geometric(1/60)
        if e == 1:
            return np.random.poisson(520)
        return np.random.poisson(800)

    def get_lambda(self):
        return np.random.randint(self.lamda_n)

    def get_h(self):
        return np.random.randint(self.h_n)

    def get_time(self):
        self.time += 0.25
        if self.time == 24:
            self.time = 0

    def check_constraints(self, m, mu, lamda):
        if mu > lamda or mu < 0:
            return False
        if isinstance(mu, complex): 
            return False
        if m * self.server_service_rate <= mu:
            return False
        return True
    def cost_delay_local_function(self, m, mu):
        if m == 0 and mu == 0: return 0
        return mu / (m * self.server_service_rate - mu)
    def cost_delay_cloud_function(self, mu, h, lamda):
        return (lamda - mu) * h 
    def cost_function(self, m, mu, h, lamda):
        return self.cost_delay_local_function(m, mu) + self.cost_delay_cloud_function(mu, h, lamda)
    # def  get_m_mu(self, action, state):
    #     if  action = 0:
    #         return [0, 0]
    #     lamd, _, h, _ = self.state
    #     opt_val = math.inf
    #     ans = [-1, -1]
    #     for m in range(1, self.max_number_of_server + 1):
    #         normalized_min_cov = self.lamda_low
    #         # coeff = [1, 1, (self.server_power_consumption * m - de_action) / self.server_power_consumption *  normalized_min_cov]
    #         # roots = np.roots(coeff)
    #         # for i in range(2):
    #         # mu = roots[i]
    #         mu = (action - self.server_power_consumption * m) * normalized_min_cov / self.server_power_consumption
    #         valid = self.check_constraints(m, mu, lamd)
    #         if valid:
    #             if self.cost_function(m, mu, h, lamd) < opt_val:
    #                 ans = [m, mu]
    #                 opt_val = self.cost_function(m, mu, h, lamd)
    #     return ans
    # power
    def get_dop(self, state):
        return self.d_sta + self.coef_dyn * state[0]
    def get_dcom(self, m, mu):
        normalized_min_cov = self.lamda_low
        return self.server_power_consumption * m + self.server_power_consumption / normalized_min_cov * mu
    
    def cal(self, action, state):
        if action == 0:
            return [0, 0]
        lamd, _, h, _ = state
        opt_val = math.inf
        ans = [-1, -1]
        for m in range(1, self.max_number_of_server + 1):
            normalized_min_cov = self.lamda_low
            # coeff = [1, 1, (self.server_power_consumption * m - de_action) / self.server_power_consumption *  normalized_min_cov]
            # roots = np.roots(coeff)
            # for i in range(2):
            # mu = roots[i]
            mu = (action - self.server_power_consumption * m) * normalized_min_cov / self.server_power_consumption
            valid = self.check_constraints(m, mu, lamd)
            if valid:
                if self.cost_function(m, mu, h, lamd) < opt_val:
                    ans = [m, mu]
                    opt_val = self.cost_function(m, mu, h, lamd)
        return ans
        # lamda, b, h, _ = state
        # d_op = self.get_dop(state)
        # if b <= d_op + 150:
        #     return [0, 0]
        # else:
        #     low_bound = 150
        #     high_bound = np.minimum(b - d_op, self.get_dcom(10, lamda))
        #     de_action = low_bound + action * (high_bound - low_bound)
        #     # print('deaction ', de_action)
        # return self.get_m_mu(action, state)

    def reward_func(self, g, d_op, d, m, mu):
        lamda, b, h, _ = self.state
        cost_delay_wireless = 0
        cost_delay = self.cost_function(m, mu, h, lamda) + cost_delay_wireless
        if d_op > b:
            cost_batery = 0
            cost_bak = self.back_up_cost_coef * d_op
        else:
            cost_batery = self.normalized_unit_depreciation_cost * np.maximum(d - g, 0)
            cost_bak = 0
        cost = cost_delay + cost_batery + cost_bak

        cost_delay_local = self.cost_delay_local_function(m, mu)
        cost_delay_cloud = self.cost_delay_cloud_function(mu, h, lamda)
        # print('\t{:20} {:20} {:20} {:10}'.format("cost_delay_local", "cost_delay_cloud", "cost_batery", "cost_bak"))
        # print('\t{:<20.3f} {:<20.2f} {:<20.2f} {:<10.2f}'.format(cost_delay_local, cost_delay_cloud, cost_batery, cost_bak))
        return cost

    def step(self, action):
        done = False
        # action = float(action)
        self.get_time()
        state = self.state
        print('\tstate: ',state)
        print('\ttime: ',self.time)
        g_t = self.get_g(state[3])
        print('\tget ', g_t)
        print('\taction: ', action)
        state = self.de_state(state)
        print('\tde_state: ',state)

        d_op = self.get_dop(state)
        number_of_server, local_workload = self.cal(action, state) 
        d_com = self.get_dcom(number_of_server, local_workload)
        d = d_op + d_com
        print('\t{:20}{:20}{:20}{:20}{:10}'.format('d_op','d_com','d','number_server','local_workload'))
        print('\t{:<20.3f}{:<20.3f}{:<20.3f}{:<20.3f}{:<10.3f}'.format(d_op, d_com, d, number_of_server, local_workload))
        reward = self.reward_func(g_t, d_op, d, number_of_server, local_workload)
        lambda_t = self.get_lambda()
        b_t = self.get_b(state, g_t, d_op, d) 
        h_t = self.get_h()
        e_t = self.get_e()
        self.state = np.array([lambda_t, b_t, h_t, e_t])
        print('\tnew state: ', self.state)
        print('\tde_new state: ', self.de_state(self.state))
        print('\tcost: ', reward)
        if b_t <= 0:
            done = True
        return self.state, 1 / reward, done, {}


















    # def cal(self, action):
    #     # opt_val = math.inf
    #     # ans = [0, 0]
    #     # ok = False
    #     # done = False
    #     for number_of_server in range(1, self.max_number_of_server + 1):
    #         local_workload = (action - number_of_server * self.server_power_consumption) / self.k
    #         if local < 0 or local_workload > self.
    #     #     coeff = [1, 1, 1 + self.server_power_consumption * number_of_server - action]
    #     #     local_workload = np.roots(coeff)[1]
    #     #     if isinstance(local_workload, complex):
    #     #         continue
    #     #     else:
    #     #         ok = True
    #     #         cost_delay_local = local_workload / (number_of_server * self.server_service_rate - local_workload)
    #     #         cost_delay_cloud = (self.state[0] - local_workload) * self.state[2]
    #     #         # print('Cost delay ' + str(cost_delay_local + cost_delay_cloud))
    #     #         if opt_val > cost_delay_local + cost_delay_cloud:
    #     #             ans = [number_of_server, local_workload]
    #     #             opt_val = cost_delay_local + cost_delay_cloud
    #     # if ok:
    #     #     return ans, done
    #     # else:
    #     #     done = True
    #     #     return [0, 0], done

    #     # number_of_server = action -
    
    # def power_model(self, action, de_state):
    #     d_dyn = self.coef_dyn * de_state[0]
    #     d_op = d_dyn + self.d_sta
    #     # params = [0, 0]
    #     params, done = self.cal(action) 
    #     number_of_server, local_workload = params
    #     # local_workload = params[1]
    #     d_com = action
    #     return d_op, d_com, d_op + d_com, number_of_server, local_workload, done

    # def get_b(self, state, action, g, d_op, d):
    #     b = state[1]
    #     if d_op > b:
    #         # print('unused batery')
    #         return b + g
    #     else:
    #         if g >= d:
    #             # print('recharge batery')
    #             return np.maximum(self.b_high, b + g - d)
    #         else:
    #             # print('discharge batery')
    #             return b + g - d
    # # constraints for delay local function

    # def cost_delay_local_function(self, m, mu):
    #     if m == 0 and mu == 0: return 0
    #     return mu / (m * self.server_service_rate - mu)
    # def cost_delay_cloud_function(self, mu, h, lamda):
    #     return (lamda - mu) * h 

    # def reward_func(self, action, g, d_op, d, number_of_server, local_workload, de_state):
    #     lamda, b = de_state[1]
    #     act = [number_of_server, local_workload]
    #     if act == [0, 0]:
    #         cost_delay_local = 0
    #     else:
    #         cost_delay_local = local_workload / (number_of_server * self.server_service_rate - local_workload)
    #     cost_delay_cloud = (self.state[0] - local_workload) * self.state[2]
    #     cost_delay_wireless = 0
    #     cost_delay = cost_delay_local + cost_delay_cloud + cost_delay_wireless
    #     if d_op > b:
    #         cost_batery = 0
    #         cost_bak = self.back_up_cost_coef * d_op
    #     else:
    #         cost_batery = self.normalized_unit_depreciation_cost * np.maximum(d - g, 0)
    #         cost_bak = 0
    #     cost = cost_delay + cost_batery + cost_bak
    #     return cost

    # def step(self, action):
    #     done = False
    #     # self.time_step += 1
    #     self.get_time()
    #     state = self.state
    #     print('\ttime: ',self.time)
    #     print('\tstate: ',state)
    #     de_state = self.de_state(state)
    #     print('\tde_state: ',state)
    #     g_t = self.get_g(state[3])
    #     print('\tget ', g_t)
    #     print('\taction: ', action)
    #     d_op, d_com, d, number_of_server, local_workload, done = self.power_model(action, de_state)
    #     # print('\t{0:10}{1:10}{2:10}{3:20}{4:10}'.format('d_op','d_com','d','number_server','local_workload'))
    #     # print('\t{0:<10.3f}{1:<10.3f}{2:<10.3f}{3:<20.3f}{4:<10.3f}'.format(d_op, d_com, d, number_of_server, local_workload))
    #     reward = self.reward_func(action, g_t, d_op, d, number_of_server, local_workload)
    #     cost_delay_local = self.cost_delay_local_function(number_of_server, local_workload)
    #     cost_delay_cloud = self.cost_delay_cloud_function(local_workload, state[2], state[0])
    #     lambda_t = self.get_lambda()
    #     b_t = self.get_b(state, action, g_t, d_op, d)
    #     h_t = self.get_h()
    #     e_t = self.get_e()
    #     self.state = np.array([lambda_t, b_t, h_t, e_t])
    #     print('\tnew state: ', self.state)
    #     print('\tde_new state: ', self.de_state(self.state))
    #     print('\treward: ', reward)
    #     if b_t < 0 or (cost_delay_cloud < 0 and cost_delay_local < 0) or reward < 0:
    #         done = True
    #         reward = 1e18
    #         self.episode += 1
    #     return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0, 1000, 0, 0])
        self.time_step = 0
        self.time = 0
        return self.state

if __name__ == '__main__':
    MyEnv = OffloadAutoscaleDiscreteEnv()
    MyEnv.reset()
    # # obs = MyEnv.observation_space.sample()
    # # print('debug: ', obs)
    # # print(MyEnv.observation_space.sample())
    for i in range(2000):
        print('STEP: ', i)
        action = 500
        state, reward, done, _ = MyEnv.step(action)
        if done:
            MyEnv.reset()

    #     print('STEP: ', i)
    #     state, reward = MyEnv.step(i+3000)
    #     # state, reward = MyEnv.step(MyEnv.state[1])
    #     # print(state)
    #     # print(reward)
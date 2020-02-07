import gym
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.optimize import minimize_scalar

class OffloadAutoscaleEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, p_coeff):
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # Wh
        self.server_service_rate = 20  # units/sec

        self.lamda_high = 100  # units/second
        self.lamda_low = 10
        self.b_high = self.batery_capacity / self.timeslot  # W
        self.b_low = 0
        self.h_high = 0.06*5  # s/unit
        self.h_low = 0.02*5
        self.e_low = 0
        self.e_high = 2
        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 15
        self.priority_coefficent = p_coeff

        # power model
        self.d_sta = 300
        self.coef_dyn = 0.5
        self.server_power_consumption = 150

        self.time_steps_per_episode = 96
        self.episode = 0

        r_high = np.array([
            self.lamda_high,
            self.b_high,
            self.h_high,
            self.e_high])
        r_low = np.array([
            self.lamda_low,
            self.b_low,
            self.h_low,
            self.e_low])
        self.observation_space = spaces.Box(low=r_low, high=r_high)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = [0, 0, 0, 0]
        self.time = 0
        self.time_step = 0

        self.d_op = 0
        self.d_com = 0
        self.d = 0
        self.m = 0
        self.mu = 0
        self.g = 0

        self.reward_time = 0
        self.reward_bak = 0
        self.reward_bat = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Transition functions
    def get_lambda(self):
        return np.random.uniform(self.lamda_low, self.lamda_high)
    def get_b(self):
        b = self.state[1]
        # print('\t', end = '')
        if self.d_op > b:
            # print('unused batery')
            return b + self.g
        else:
            if self.g >= self.d:
                # print('recharge batery')
                return np.minimum(self.b_high, b + self.g - self.d)
            else:
                # print('discharge batery')
                return b + self.g - self.d
    def get_h(self):
        return np.random.uniform(self.h_low, self.h_high)
    def get_e(self):
        if self.time >= 9 and self.time < 15:
            return 2
        if self.time < 6 or self.time >= 18:
            return 0
        return 1


    def get_time(self):
        self.time += 0.25
        if self.time == 24:
            self.time = 0
    def get_g(self):
        e = self.state[3]
        if e == 0:
            return np.random.exponential(60) + 100
            # return np.random.normal(200,100)
        if e == 1:
            return np.random.normal(520, 130)
            # return np.random.normal(400, 100)
        return np.random.normal(800, 95)
        # return np.random.normal(600, 100)

    def check_constraints(self, m, mu):
        if mu > self.state[0] or mu < 0: return False
        if isinstance(self.mu, complex): return False
        if m * self.server_service_rate <= mu: return False
        return True
    def cost_delay_local_function(self, m, mu):
        if m == 0 and mu == 0: return 0
        return mu / (m * self.server_service_rate - mu)
    def cost_delay_cloud_function(self, mu, h, lamda):
        return (lamda - mu) * h
    def cost_function(self, m, mu, h, lamda):
        return self.cost_delay_local_function(m, mu) + self.cost_delay_cloud_function(mu, h, lamda)
    def  get_m_mu(self, de_action):
        lamd, _, h, _ = self.state
        opt_val = math.inf
        ans = [-1, -1]
        for m in range(1, self.max_number_of_server + 1):
            # coeff = [1, 1, (self.server_power_consumption * m - de_action) / self.server_power_consumption *  normalized_min_cov]
            # roots = np.roots(coeff)
            # for i in range(2):
            # mu = roots[i]

            normalized_min_cov = self.lamda_low
            mu = (de_action - self.server_power_consumption * m) * normalized_min_cov / self.server_power_consumption
            valid = self.check_constraints(m, mu)
            if valid:
                if self.cost_function(m, mu, h, lamd) < opt_val:
                    ans = [m, mu]
                    opt_val = self.cost_function(m, mu, h, lamd)
        return ans
    # power
    def get_dop(self):
        return self.d_sta + self.coef_dyn * self.state[0]
    def get_dcom(self, m, mu):
        normalized_min_cov = self.lamda_low
        return self.server_power_consumption * m + self.server_power_consumption / normalized_min_cov * mu
        # return self.server_power_consumption * m

    def cal(self, action):
        lamda, b, h, _ = self.state
        d_op = self.get_dop()
        if b <= d_op + 150:
            return [0, 0]
        else:
            low_bound = 150
            high_bound = np.minimum(b - d_op, self.get_dcom(self.max_number_of_server,lamda))
            de_action = low_bound + action * (high_bound - low_bound)
            # print('deaction ', de_action)
            return self.get_m_mu(de_action)

    def reward_func(self, action):
        lamda, b, h, _ = self.state
        cost_delay_wireless = 0
        self.m, self.mu = self.cal(action)
        cost_delay = self.cost_function(self.m, self.mu, h, lamda) + cost_delay_wireless
        if self.d_op > b:
            cost_batery = 0
            cost_bak = self.back_up_cost_coef * self.d_op
        else:
            cost_batery = self.normalized_unit_depreciation_cost * np.maximum(self.d - self.g, 0)
            cost_bak = 0
        cost_bak = cost_bak * self.priority_coefficent
        cost_batery = cost_batery * self.priority_coefficent
        cost_delay = cost_delay * (1 - self.priority_coefficent)
        self.reward_bak = cost_bak
        self.reward_bat = cost_batery
        self.reward_time = cost_delay
        cost = cost_delay + cost_batery + cost_bak
        # cost_delay_local = self.cost_delay_local_function(self.m, self.mu)
        # cost_delay_cloud = self.cost_delay_cloud_function(self.mu, h, lamda)
        # print('\t{:20} {:20} {:20} {:10}'.format("cost_delay_local", "cost_delay_cloud", "cost_batery", "cost_bak"))
        # print('\t{:<20.3f} {:<20.2f} {:<20.2f} {:<10.2f}'.format(cost_delay_local, cost_delay_cloud, cost_batery, cost_bak))
        return cost

    def step(self, action):
        done = False
        action = float(action)
        self.get_time()
        state = self.state
        # print('time_step: ', self.time_step)
        self.time_step += 1
        # print('\tstate: ',state)
        # print('\ttime: ',self.time)
        self.g = self.get_g()
        # print('\tget ', g_t)
        # print('\taction: ', action)

        self.d_op = self.get_dop()
        self.m, self.mu = self.cal(action)
        self.d_com = self.get_dcom(self.m, self.mu)
        self.d = self.d_op + self.d_com
        # print('\t{:20}{:20}{:20}{:20}{:10}'.format('d_op','d_com','d','number_server','local_workload'))
        # print('\t{:<20.3f}{:<20.3f}{:<20.3f}{:<20.3f}{:<10.3f}'.format(d_op, d_com, d, number_of_server, local_workload))
        reward = self.reward_func(action)
        lambda_t = self.get_lambda()
        b_t = self.get_b()
        h_t = self.get_h()
        e_t = self.get_e()
        self.state = np.array([lambda_t, b_t, h_t, e_t])
        # print('\tnew state: ', self.state)
        # print('\tcost: ', reward)
        if  self.time_step >= self.time_steps_per_episode:
            done = True
            self.episode += 1
        return self.state, 1 / reward, done, {}

    def reset(self):
        self.state = np.array([self.lamda_low, self.b_high, self.h_low, self.e_low])
        self.time = 0
        self.time_step = 0
        return self.state
    def render(self):
        # print('{:>7} {:>7} {:>7} {:>7} {:>4} {:>7} {:>7} {:>7} {:>4} {:>4}'.format("g", "d_op", "d_com", "d", "m", "mu", "lamd_t+1","b_t+1", "h_t+1", "e_t+1"))
        # print('{:7.2f} {:7.2f} {:7.2f} {:7.2f} {:4} {:7.2f} {:8.2f} {:7.2f} {:5.2f} {:5.0f}'.format(self.g,self.d_op, self.d_com,self.d,self.m,self.mu, self.state[0],self.state[1],self.state[2],self.state[3]))
        # return self.state[0],self.state[1],self.state[2],self.state[3],self.g,self.d_op, self.d_com,self.d,self.m,self.mu
        return  self.reward_time, self.reward_bak, self.reward_bat
    def fixed_action_cal(self, fixed_action):
        lamda, b, h, _ = self.state
        d_op = self.get_dop()
        low_bound = 150
        high_bound = np.minimum(b - d_op, self.get_dcom(self.max_number_of_server,lamda))
        if high_bound < low_bound:
            return 0
        if fixed_action < low_bound:
            return 0
        if fixed_action > high_bound:
            return 1
        else:
            return (fixed_action-low_bound)/(high_bound-low_bound)
    def myopic_action_cal(self):
        lamda, b, h, _ = self.state
        d_op = self.get_dop()
        if b <= d_op + 150:
            return 0
        else:
            ans = math.inf
            for m in range(1, self.max_number_of_server):
                def f(mu, m, h, lamda):
                    return (1 - self.priority_coefficent) * (mu/(m*self.server_service_rate-mu)+h*(lamda - mu))+ self.priority_coefficent * (self.normalized_unit_depreciation_cost*(self.server_power_consumption*m+self.server_power_consumption/self.lamda_low*mu))
                res = minimize_scalar(f, bounds=(0, min(lamda,m*self.server_service_rate)), args=((m, h, lamda)), method='bounded')
                if res.fun < ans:
                    ans = res.fun
                    params = [m, res.x]
            d_com = self.server_power_consumption*params[0]+self.server_power_consumption/self.lamda_low*params[1]
            return self.fixed_action_cal(d_com)
# MyEnv = OffloadAutoscaleEnv()
# MyEnv.reset()
# MyEnv.render()
# # # # state_list = []
# for i in range(20):
#     print('STEP: ', i)
#     action = MyEnv.myopic_action_cal()
#     print(action)
# # # #     action = MyEnv.action_space.sample()
#     state, reward, done, info = MyEnv.step(action)
#     MyEnv.render()
#     state_list.append(MyEnv.render()[4])
#     if done: MyEnv.reset()
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# sns.set(style='ticks')
# df=pd.DataFrame({'x': range(200*4), 'y_1': state_list})
#  # 'y_2': avg_rewards_random, 'y_3': avg_rewards_fixed_0, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2})
# # plt.xlabel("Time Slot")
# # # plt.ylabel("Batery")
# # plt.ylabel("Number Servers")
# # plt.scatter( 'x', 'y_1', data=df, marker='o', color='skyblue', linewidth=0.1, label="m")
# plt.plot( 'x', 'y_1', data=df, marker='', color='green', linewidth=1, label="g")
# # plt.hist(state_list,bins = 20*8)
# # sns.kdeplot(state_list);
# plt.legend()
# plt.show()
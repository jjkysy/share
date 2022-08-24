"""

fixed final state finite horizon LQR

author: Shengyue Yao

"""

import copy
import numpy as np
from numpy import arange
import scipy.linalg as la
import Globalvar
import math
# import pdb
# import time
# import numba as nb


class LQRPlanner:

    def __init__(self, ss, gs, ts, rand_area=None):
        self.MAX_ITER = Globalvar.search_leader
        self.ss_all = ss
        self.gs_all = gs
        self.ss_original = np.array([ss.x, ss.y, ss.v, ss.theta]).reshape(4, 1)
        self.gs_original = np.array([gs.x, gs.y, gs.v, gs.theta]).reshape(4, 1)
        self.ts = ts
        self.rand_area = rand_area
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.R = np.array([[Globalvar.r_alpha, 0], [0, Globalvar.r_omega]])
        # self.gram_vector = np.vectorize(self.grammian, excluded='self', otypes=[np.ndarray])
        # self.delta_vector = np.vectorize(self.delta, excluded='self', otypes=[np.ndarray])

    def lqr_tau(self):
        # input start s, end s
        # 1. calculate grammian function with tau
        # 2. calculate d(tau) function with tau
        # 3. find the best tau by partial derivitive
        # 4. get the optimal cost for each od pair
        # 1-4 could write in one function
        # 5. return the function of control
        # 6. return the function of optimal state sequence
        # pdb.set_trace()
        # time0 = time.time()
        # time1 = time.time()
        tau, cost, G, d, s_hat = self.opt_cost()
        # time2 = time.time()
        # print(time2 - time1)
        # pdb.set_trace()
        # time1 = time.time()
        return cost, tau, G, d, s_hat
        # if shortest == 1:
        #     return self.generate_new(tau, d, G, s_hat)
        # if shortest == 2:  # maybe change later to avoid double calculate
        #     # not finding the new node, but rewire links in step 6 or 7
        #     # time5 = time.time()
        #     return self.generate_all(tau, d, cost)

    def generate_new(self, tau, d, G, s_hat):
        # node: state, x, y, v, theta
        # controller, a, omega
        # path state, path_x, path_y, path_v, path_theta
        # path controller, path_a, path_omega
        # t_label, cost, parent
        # pdb.set_trace()
        # time3 = time.time()
        s_new_linear = self.opt_s(d, tau, self.ts)  # with vx, vy
        initial_u_linear = self.opt_u(tau, 0, d)  # with ax, ay
        new_node = copy.deepcopy(self.ss_all)
        # pdb.set_trace()
        s_new = self.transform_s(s_new_linear, 2)
        initial_u = self.transform_u(self.ss_original, initial_u_linear)
        new_node.x = float(s_new[0])
        new_node.y = float(s_new[1])
        new_node.v = float(s_new[2])
        new_node.theta = float(s_new[3])

        new_node.parent = self.ss_all

        new_node.path_x = [float(self.ss[0]), float(s_new[0])]
        new_node.path_y = [float(self.ss[1]), float(s_new[1])]
        new_node.path_v = [float(self.ss_original[2]), float(s_new[2])]
        new_node.path_theta = [float(self.ss_original[3]), float(s_new[3])]
        new_node.path_a = [float(initial_u[0]), 0]
        new_node.path_omega = [float(initial_u[1]), 0]

        new_node.t_label = self.ss_all.t_label + self.ts
        cost_one_step = self.cost(self.ts, s_hat, G)
        new_node.cost = self.ss_all.cost + cost_one_step
        return new_node

    def generate_all(self, tau, d, cost=0):
        # not finding the new node, but rewire links in step 6 or 7
        # time5 = time.time()
        t = arange(0, tau+self.ts, self.ts)
        u_linear_list = self.opt_u(tau, t, d)  # return 3d ndarry
        s_linear_list = self.opt_s(d, tau, t)  # return list
        # time6 = time.time()
        s_list = list(map(self.transform_s, s_linear_list))
        u_list = list(map(self.transform_u, s_list, u_linear_list))
        # pdb.set_trace()
        # time3 = time.time()
        # time4 = time.time()
        # add the start state
        # pdb.set_trace()
        u_list[-1] = np.array([0, 0]).reshape(2, 1)
        # if self.ss_all.t_label == 22: pdb.set_trace()
        # if self.rand_area is not None:
        #     cost = self.get_eq_cost(s_linear_list, u_linear_list, tau)
        # time7= time.time()
        x_list, y_list, v_list, theta_list, a_list, omega_list = self.extract_seq(s_list, u_list)
        # time8 = time.time()
        lqr_result = (cost, tau, x_list, y_list, v_list, theta_list, a_list, omega_list)
        # time9 = time.time()
        # print('timemode2:{},{},{},{}'.format(time6-time5,time7-time6, time8-time7,time9-time8))
        # pdb.set_trace()
        # print(time1-time0,time2-time1,time3-time2,time4-time2, time3-time7)
        return lqr_result

    @staticmethod
    def extract_seq(s, u):
        x_list = list(map(lambda x: float(x[0]), s))  # numpy slice
        y_list = list(map(lambda x: float(x[1]), s))
        v_list = list(map(lambda x: float(x[2]), s))
        theta_list = list(map(lambda x: float(x[3]), s))
        a_list = list(map(lambda x: float(x[0]), u))
        omega_list = list(map(lambda x: float(x[1]), u))
        return x_list, y_list, v_list, theta_list, a_list, omega_list

    def opt_u(self, tau, t, d):
        A = self.A
        B = self.B
        R = self.R
        try:
            t[0]
        except TypeError:
            return la.inv(R) @ B.T @ la.expm(A.T * (tau - t)) @ d
        else:
            mu = np.einsum('jk,i->ijk', A.T, tau-t)
            # pdb.set_trace()
            return la.inv(R) @ B.T @ (np.eye(4) + mu) @ d

    def opt_s(self, d, tau, t):
        A = self.A
        B = self.B
        R = self.R
        # return matrix of states, dim: 4x1
        part_1 = np.concatenate((np.eye(4), np.zeros([4, 4])), axis=1)
        part_2 = np.concatenate((A, np.zeros([4, 4])), axis=0)
        part_3 = np.concatenate((B @ la.inv(R) @ B.T, -A.T), axis=0)
        part_4 = np.concatenate((part_2, part_3), axis=1)
        part_5 = np.concatenate((self.gs, d), axis=0)
        # pdb.set_trace()
        try:
            t[0]
        except TypeError:
            return part_1 @ (la.expm(part_4 * (t - tau)) @ part_5)
        else:
            return list(map(lambda a: part_1@(la.expm(part_4*(a-tau)))@part_5, t))

    def opt_cost(self):
        # pdb.set_trace()
        t = np.arange(self.ts, self.MAX_ITER + self.ts, self.ts)
        l_G_f = list(map(self.grammian, t))
        l_hat = list(map(self.s_hat, t))
        l_d_f = list(map(self.delta, l_hat, l_G_f))
        l_residual = list(map(self.residual, l_d_f))
        # pdb.set_trace()
        least_id = l_residual.index(min(l_residual))
        # pdb.set_trace()
        G = l_G_f[least_id]
        d = l_d_f[least_id]
        s_hat = l_hat[least_id]
        tau = (least_id + 1) * self.ts
        # pdb.set_trace()
        # if self.rand_area is not None:
        #     result = self.generate_all(tau, d)
        #     cost = result[0]
        else:
            cost = self.cost(tau, s_hat, G)
        return tau, cost, G, d, s_hat

    def cost(self, t, s_hat, G):
        return float(t + (self.gs - s_hat).T @ la.inv(G) @ self.grammian(t) @ la.inv(G) @ (self.gs - s_hat))

    def residual(self, d):
        return abs(1 - d.T @ self.B @ la.inv(self.R) @ self.B.T @ d)

    def grammian(self, tau):
        R = self.R
        g1, g2 = float(R[0, 0]), float(R[1, 1])
        # pdb.set_trace()
        g00 = tau ** 3 / (3 * g1)
        g11 = tau ** 3 / (3 * g2)
        g02 = tau ** 2 / (2 * g1)
        g13 = tau ** 2 / (2 * g2)
        g22 = tau / g1
        g33 = tau / g2
        gram = np.array([[g00, 0, g02, 0], [0, g11, 0, g13], [g02, 0, g22, 0], [0, g13, 0, g33]])
        return gram

    def delta(self, s_hat, gram):
        return la.inv(gram) @ (self.gs - s_hat)
        # pdb.set_trace()

    def s_hat(self, tau):
        return la.expm(self.A * tau) @ self.ss

    # @staticmethod
    # def transform_u(s: np.ndarray, u: np.ndarray, _=2) -> np.ndarray:
    #     v, theta = float(s[2]), float(s[3])
    #     if _ == 1:  # from alpha, lambda to ax, ay
    #         m_trans = np.array([[np.cos(theta), -v*np.sin(theta)], [np.sin(theta), v*np.cos(theta)]])
    #         return m_trans @ u
    #     elif _ == 2:  # ax ay, to alpha, lambda
    #         m_trans = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta) / v, np.cos(theta) / v]])
    #         # pdb.set_trace()
    #         return m_trans @ u
    #
    # @staticmethod
    # def transform_s(s: np.ndarray, _=2) -> np.ndarray:
    #     s1, s2 = float(s[2]), float(s[3])  # s1:vx or v, s2:vy or theta
    #     if _ == 1:  # from v, theta to vx, vy
    #         vx = s1 * np.cos(s2)
    #         vy = s1 * np.sin(s2)
    #         return np.array([float(s[0]), float(s[1]), vx, vy]).reshape(4, 1)
    #     elif _ == 2:  # inverse transform
    #         v = math.hypot(s1, s2)
    #         theta = np.arccos(s1 / v)
    #         if s2 < 0:
    #             theta = 2 * np.pi - theta
    #         return np.array([float(s[0]), float(s[1]), v, theta]).reshape(4, 1)

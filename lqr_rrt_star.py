"""

Path planning code with C-RRT*

author: Shengyue Yao

"""
# import multiprocessing

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
# import time
import itertools
# from collections import deque

from LQRplanner import LQRPlanner
from LQR_follow import LQRF
import pdb
import geometry as geo
import Globalvar

show_animation = True


# import os
# import multiprocessing
# import os


class LQRRRTStar:
    """
    Class for RRT star planning with LQR planning
    """

    class Result:

        def __init__(self, vid: int, path: list, path_v: list, path_theta: list, path_a: list, path_omega: list,
                     final_cost: float, final_tau: float, reach: int):
            self.vid = vid
            self.path = path
            self.path_v = path_v
            self.path_theta = path_theta
            self.path_a = path_a
            self.path_omega = path_omega
            self.final_cost = final_cost
            self.final_tau = final_tau
            self.reach = reach

    def __init__(self, g_road, od_list, vid, rand_area,
                 goal_sample_rate=Globalvar.goal_sample_rate,
                 max_iter=50000,
                 step_size=Globalvar.dt,  # might be 0.5 or smaller
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        g_raod:obstacles geometry
        goal_sample_rate:rate of puting samplled point at destination
        connect_circle_dist:neighbor search area in step 5
        step_size: 1 second
        """
        self.vid = vid
        self.leader = 1
        self.geo = geo.Geometry(g_road, od_list)
        start = self.geo.start(vid)
        goal = self.geo.end()
        self.start = geo.Node(start[0], start[1], start[2], start[3])  # start, end = node(x,y,v,theta)
        self.end = geo.Node(goal[0], goal[1], goal[2], goal[3])
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.g_road = g_road  # shapefile
        self.connect_circle_dist = 30 * Globalvar.lroad / (Globalvar.dt * 100)
        # parameter for near nodes circle
        self.node_list = []
        self.goal_xy_th = Globalvar.tol  # buffer to reach goal in meters
        self.step_size = step_size
        self.rand_area = rand_area  # for follower, sample from expected tracks
        if rand_area:
            self.eq_location = rand_area.location
        else:
            self.eq_location = None
        # self.pre_track = pre_location  # previous optimal path
        self.desire_speed = Globalvar.desire_speed
        self.gen_leader = Globalvar.gen_leader
        self.change = 0
        # pdb.set_trace()

    def planning(self, animation=True, search_until_max_iter=True):
        """
        RRT Star planning

        animation: flag for animation on or off
        """
        # pdb.set_trace()
        self.node_list = [self.start]
        new_node = None
        # pool = multiprocessing.Pool(3)
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            print("change:", self.change)
            # pdb.set_trace()
            # if i == 30: pdb.set_trace()
            # if -x_min + self.start.x >= 200:
            #     time_200 = time.time()
            #     print('time for 200 m:%f' % (time_200-time_init))
            #     pdb.set_trace()
            # step 2
            if self.rand_area:
                rnd = self.geo.rand_p_f(self.goal_sample_rate, self.rand_area)
                self.leader = 0
                self.end = geo.Node(self.rand_area.location[-1][0], self.rand_area.location[-1][1],
                                    self.rand_area.v[-1], self.rand_area.theta[-1])
                # if t_max>=32 and rnd.t_label>32: pdb.set_trace()
            # else:
            #     # pdb.set_trace()
            #     x_list = [i.x for i in self.node_list]
            #     x_min = min(x_list)
            #     rnd = self.geo.rand_p(self.goal_sample_rate, x_min, self.vid)
            #     # pdb.set_trace()
            # pdb.set_trace()
            # step 3
            if self.leader == 1:
                # time_steer1 = time.time()
                # if i==447:pdb.set_trace()
                nearest_node, cost, tau, G, d, s_hat = self.get_nearest_node(rnd)  # return a index, use lqr
                # timep = time.time()
                # print('search_near:{}'.format(timep - time_steer1))
                if nearest_node is not None:
                    # if i == 46: pdb.set_trace()
                    # pdb.set_trace()
                    new_node = self.steer(nearest_node, rnd, 1, tau, d, G, s_hat)
                    # pdb.set_trace()
                    if self.node_duplicate(new_node):  # check later the efficiency
                        continue
                    # time_steer2 = time.time()
                    # time_step_3 = time_steer2 - time_steer1
                    # print('time step 3 is %f' % time_step_3)
                else:
                    continue
            # else:
            #     # time_steer1 = time.time()
            #     # if self.vid == 3 and rnd.v<10 and rnd.t_label<3: pdb.set_trace()
            #     # if rnd.v == self.rand_area.v[int(rnd.t_label/Globalvar.dt)]:
            #     nearest_node, min_cost, s_list, u_list, cost_list = self.get_nearest_node_follow(rnd)
            #     if min_cost < 0.001 and (not search_until_max_iter):
            #         # pdb.set_trace()
            #         # terminate the process if one equilibrium state is merged
            #         last_index = self.generate_rest_node(nearest_node)
            #         return self.final_all(last_index)
            #     if nearest_node is not None:
            #         new_node = self.steer_follow(nearest_node, rnd, 1, s_list, u_list, cost_list)
            #         # print('r_v:{}, r_theta: {}'.format(rnd.v, rnd.theta))
            #         # print('t:{},v:{},theta:{},a:{},omega:{},near:{}'.format(rnd.t_label, new_node.path_v,
            #         #                                                         new_node.path_theta, new_node.path_a,
            #         #                                                         new_node.path_omega, self.node_list.index(nearest_node)))
            #         # if self.vid == 3 and rnd.v<10: pdb.set_trace()
            #     # else:
            #     #     nearest_node, cost, tau, G, d, s_hat = self.get_nearest_node(rnd)  # return a index, use lqr
            #     #     if nearest_node is not None:
            #     #         new_node = self.steer(nearest_node, rnd, 1, tau, d, G, s_hat)

                # if self.node_duplicate(new_node):
                #     continue
                #     # time_steer2 = time.time()
                #     # time_step_3 = time_steer2 - time_steer1
                #     # print('time step 3 is %f' % time_step_3)
            if new_node is None:
                continue
            # step 4
            # pdb.set_trace()
            if self.geo.check_within(new_node, self.eq_location, self.vid):
                # pdb.set_trace()
                # print('x,{},y,{},v,{},theta,{}'.format(new_node.x, new_node.y, new_node.v, new_node.theta))
                new_node.parent.a = new_node.parent.path_a[-1] = float(new_node.path_a[0])
                new_node.parent.omega = new_node.parent.path_omega[-1] = float(new_node.path_omega[0])
                # step 5
                near_nodes = self.find_near_nodes(new_node)
                # step 6
                # time_parent1 = time.time()
                # pdb.set_trace()
                self.choose_parent(new_node, near_nodes)
                # time_parent2 = time.time()
                # time_step_6 = time_parent2 - time_parent1
                # print('time step 6 is %f' % time_step_6)
                # pdb.set_trace()
                self.node_list.append(new_node)
                # step 7
                # pdb.set_trace()
                # time_rewire1 = time.time()
                self.rewire(new_node, near_nodes)
                # time_rewire2 = time.time()
                # time_step_7 = time_rewire2 - time_rewire1
                # print('time step 7 is %f' % time_step_7)
            # debug
            # for n in self.node_list:
            #     if 0 in n.path_a:
            #         pdb.set_trace()
            # step 8
            if not search_until_max_iter:  # check reaching the goal without max iteration
                # if i > 1000: pdb.set_trace()
                last_index = self.search_best_goal_node()
                if last_index:
                    if self.leader == 0:
                        if abs(self.node_list[last_index].t_label - self.rand_area.t) <= self.step_size:
                            return self.final_all(last_index)
                        else:
                            continue
                        # pdb.set_trace()
                    else:
                        return self.final_all(last_index)

            if animation and i % 10 == 0:
                self.draw_graph(rnd)
                # pdb.set_trace()
            # pdb.set_trace()
        print("reached max iteration")
        # step 9
        # pdb.set_trace()
        last_index = self.search_best_goal_node()
        if last_index:
            # pdb.set_trace()
            return self.final_all(last_index)
        else:
            print("Cannot find path")
        return None

    def get_nearest_node(self, rnd):
        # pdb.set_trace()
        dlist = list(map(self.calc_dist, self.node_list, itertools.repeat(rnd, len(self.node_list))))  # don't need list, avoid ask length, zip(map())
        near_node_dist = [self.node_list[i] for i in range(len(dlist)) if dlist[i] <= self.desire_speed *
                          self.step_size * self.gen_leader]
        # near_node_dist = self.node_list
        if near_node_dist:
            node_size = len(near_node_dist)
            # return cost tau G d s_hat
            # result = [self.steer(near_node_dist[i], rnd) for i in range(node_size)]
            result = list(map(self.steer, near_node_dist, itertools.repeat(rnd, node_size)))  # comprehension
            costlist = list(map(lambda x: x[0], result))
            # for i in near_inds_dist:
            #     costlist.append(self.steer(self.node_list[i], rnd, 0))  # index of node in node_list
            if min(costlist):
                minind_cost = costlist.index(min(costlist))
                min_node = near_node_dist[minind_cost]
                cost, tau, G, d, s_hat = result[minind_cost]
                return min_node, cost, tau, G, d, s_hat
            else:
                return None
        else:
            minind = dlist.index(min(dlist))
            min_node = self.node_list[minind]
            cost, tau, G, d, s_hat = self.steer(min_node, rnd)
            return min_node, cost, tau, G, d, s_hat

    def steer(self, from_node, to_node, hs=0, tau=None, d=None, G=None, s_hat=None, cost=None):
        # if it is the shortest,
        ts = self.step_size
        lqr = LQRPlanner(from_node, to_node, ts, self.rand_area)
        if hs == 0:
            return lqr.lqr_tau()
        elif hs == 1:
            # pdb.set_trace()
            return lqr.generate_new(tau, d, G, s_hat)
        else:
            # pdb.set_trace()
            return lqr.generate_all(tau, d, cost)

    def steer_follow(self, from_node, to_node, hs=0, s_list=None, u_list=None, cost_list=None):
        ts = self.step_size
        lqr =  LQRF(from_node, to_node, ts, self.rand_area)
        if hs == 0:
            return lqr.lqr_planning()
        elif hs == 1:
            # pdb.set_trace()
            return lqr.generate_new_follow(s_list, u_list, cost_list)

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list) + 1
        dim = 4
        # r = self.connect_circle_dist
        r = self.connect_circle_dist * np.power(np.log(n_node) / n_node, 1 / dim)
        # pdb.set_trace()
        print("r value:", r)
        # pdb.set_trace()
        # may save one iteration
        # if expand_dist exists, search vertices in a range no more than expand_dist
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_nodes = [self.node_list[i] for i in range(len(dist_list)) if dist_list[i] <= r ** 2]
        parent = new_node.parent
        try:
            near_nodes.remove(parent)
        except ValueError:
            pass
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return None
        # search nearest cost in near_inds
        # pdb.set_trace()
        costs = new_node.cost - new_node.parent.cost
        node_size = len(near_nodes)
        # pdb.set_trace()
        lqr_result = list(map(self.steer, near_nodes, itertools.repeat(new_node, node_size)))
        costlist = list(map(lambda x: x[0], lqr_result))
        sorted_index = np.argsort(costlist)
        # pdb.set_trace()
        for i in sorted_index:
            if costlist[i] < costs:
                cost, tau, G, d, s_hat = lqr_result[i]
                near_node = near_nodes[i]

                path_result = self.steer(near_node, new_node, 2, tau, d, cost=cost)
                l_cost, tau, x_list, y_list, v_list, theta_list, a_list, omega_list = path_result
                change_node = copy.deepcopy(new_node)
                change_node.path_x = x_list
                change_node.path_y = y_list
                change_node.path_v = v_list
                change_node.path_theta = theta_list
                change_node.path_a = a_list
                change_node.path_omega = omega_list
                change_node.t_label = near_node.t_label + tau
                if self.geo.check_within(change_node, self.eq_location, self.vid):
                    # node: state, x, y, v, theta
                    # controller, a, omega
                    # path state, path_x, path_y, path_v, path_theta
                    # path controller, path_a, path_omega
                    # t_label, cost, parent
                    # pdb.set_trace()
                    del change_node
                    new_node.parent = near_node
                    new_node.t_label = near_node.t_label + tau
                    new_node.cost = l_cost + near_node.cost
                    new_node.parent.a = new_node.parent.path_a[-1] = a_list[0]
                    new_node.parent.omega = new_node.parent.path_omega[-1] = omega_list[0]

                    new_node.path_x = x_list
                    new_node.path_y = y_list
                    new_node.path_v = v_list
                    new_node.path_theta = theta_list
                    new_node.path_a = a_list
                    new_node.path_omega = omega_list
                    self.change += 1
                    return
                else:
                    del change_node
                    continue
            else:
                return None

    def rewire(self, new_node, near_nodes):
        if not near_nodes:
            return None
        new_node_cost = new_node.cost
        node_size = len(near_nodes)
        lqr_result = list(map(self.steer, itertools.repeat(new_node, node_size), near_nodes))
        costlist = list(map(lambda x: x[0], lqr_result))
        sorted_index = np.argsort(costlist)
        for i in sorted_index:
            near_node = near_nodes[i]
            new_cost = costlist[i] + new_node_cost
            if new_cost < near_node.cost:
                cost, tau, G, d, s_hat = lqr_result[i]

                path_result = self.steer(new_node, near_node, 2, tau, d, cost=cost)
                l_cost, tau, x_list, y_list, v_list, theta_list, a_list, omega_list = path_result
                change_node = copy.deepcopy(near_node)
                change_node.path_x = x_list
                change_node.path_y = y_list
                change_node.path_v = v_list
                change_node.path_theta = theta_list
                change_node.path_a = a_list
                change_node.path_omega = omega_list
                change_node.t_label = new_node.t_label + tau
                if self.geo.check_within(change_node, self.eq_location, self.vid):
                    del change_node
                    near_node.parent = new_node
                    near_node.t_label = new_node.t_label + tau
                    near_node.cost = new_cost
                    near_node.parent.a = near_node.parent.path_a[-1] = a_list[0]
                    near_node.parent.omega = near_node.parent.path_omega[-1] = omega_list[0]

                    near_node.path_x = x_list
                    near_node.path_y = y_list
                    near_node.path_v = v_list
                    near_node.path_theta = theta_list
                    near_node.path_a = a_list
                    near_node.path_omega = omega_list
                    self.change += 1
                    return
                else:
                    del change_node
                    continue
            else:
                return None

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist(n, self.end) for n in self.node_list]
        goal_ids = [i for i in range(len(dist_to_goal_list)) if dist_to_goal_list[i] <= self.goal_xy_th]
        # a list of all ids fulfilled the termination requirement
        # pdb.set_trace()
        print('min dist to goal {}'.format(min(dist_to_goal_list)))
        if not goal_ids:
            return None
        cost_list = [self.node_list[i].cost for i in goal_ids]
        # a list of all the cost of fulfilled ids
        min_id = goal_ids[cost_list.index(min(cost_list))]  # terminal state id
        # node = self.node_list[min_id]
        # while node.parent:
        #     if self.geo.check_geometry(node):
        #         continue
        #     else:
        #         return None
        return min_id

    def final_all(self, goal_index):
        s_end = np.array([self.end.x, self.end.y, self.end.v, self.end.theta]).reshape(4, 1)
        node = self.node_list[goal_index]
        goal_end = np.array([node.x, node.y, node.v, node.theta]).reshape(4, 1)
        reach = int((s_end == goal_end).all())
        path = self.generate_final_course(goal_index)
        path_v = self.generate_final_v(goal_index)
        path_theta = self.generate_final_theta(goal_index)
        path_a = self.generate_final_a(goal_index)
        path_omega = self.generate_final_omega(goal_index)
        final_cost = node.cost
        final_tau = node.t_label
        result = self.Result(self.vid, path, path_v, path_theta, path_a, path_omega, final_cost, final_tau, reach)
        # pdb.set_trace()
        return result

    def generate_final_course(self, goal_index):
        print("final")
        path = []
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x[1:]), reversed(node.path_y[1:])):
                path.append([ix, iy])
            # print(node.path_x, node.t_label)
            # pdb.set_trace()
            node = node.parent
        path.append([self.start.x, self.start.y])
        path.reverse()
        return path

    def generate_final_v(self, goal_index):
        path_v = []
        node = self.node_list[goal_index]
        while node.parent:
            for iv in reversed(node.path_v[1:]):
                path_v.append(iv)
            node = node.parent
        path_v.append(self.start.v)
        path_v.reverse()
        return path_v

    def generate_final_theta(self, goal_index):
        path_theta = []
        node = self.node_list[goal_index]
        while node.parent:
            for itheta in reversed(node.path_theta[1:]):
                path_theta.append(itheta)
            node = node.parent
        path_theta.append(self.start.theta)
        path_theta.reverse()
        return path_theta

    def generate_final_a(self, goal_index):
        path_a = []
        # pdb.set_trace()
        node = self.node_list[goal_index]
        while node.parent:
            # if node == self.node_list[goal_index]:
            #     a_extract = node.path_a[1:]
            #     a_extract.reverse()
            #     try:
            #         (self.end.v ** 2 - node.v ** 2) / (2 * self.calc_dist(node, self.end))
            #     except ZeroDivisionError:
            #         last_a = 0
            #     else:
            #         last_a = (self.end.v ** 2 - node.v ** 2) / (2 * self.calc_dist(node, self.end))
            #     a_extract[0] = last_a
            #     path_a += a_extract
            # else:
            for ia in reversed(node.path_a[1:]):
                path_a.append(ia)
            node = node.parent
        path_a.append(self.start.a)
        path_a.reverse()
        return path_a

    def generate_final_omega(self, goal_index):
        path_omega = []
        node = self.node_list[goal_index]
        while node.parent:
            # if node == self.node_list[goal_index]:
            #     omega_extract = node.path_omega[1:]
            #     omega_extract.reverse()
            #     last_omega = (node.parent.omega - node.omega)/self.step_size
            #     omega_extract[0] = last_omega
            #     path_omega += omega_extract
            # else:
            for iomega in reversed(node.path_omega[1:]):
                path_omega.append(iomega)
            node = node.parent
        path_omega.append(self.start.omega)
        path_omega.reverse()
        return path_omega

    @staticmethod
    def calc_dist(from_node, to_node):
        dx = from_node.x - to_node.x
        dy = from_node.y - to_node.y
        return math.hypot(dx, dy)

    def node_duplicate(self, node):
        dup = list(map(lambda i: (i.x, i.y, i.v, i.theta) == (node.x, node.y, node.v, node.theta), self.node_list))
        if True in dup:
            return True
        else:
            return False

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #                              lambda event: [exit(0) if event.key == 'escape' else None])
        minx, miny, maxx, maxy = self.g_road.bounds
        minx = self.end.x
        maxx = self.start.x
        plt.xlim([minx - 10, maxx + 10])
        plt.ylim([miny - 10, maxy + 10])
        plt.plot(*self.g_road.exterior.xy, 'r')
        plt.plot(self.start.x, self.start.y, 'oy')
        plt.plot(self.end.x, self.end.y, 'or')
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^g")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-k")
        plt.grid(True)
        plt.pause(0.01)

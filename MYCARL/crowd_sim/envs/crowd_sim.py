import logging
from time import time
import gym
import matplotlib.lines as mlines
import numpy as np
from numpy.lib.arraysetops import isin
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.state import FullState
from scipy.stats import norm as norm_pdf
from numba import jit, cuda
import math
import cv2
from math import atan2, pi

table_size = 10000
table_offset = 5000
scale = 1000
# 标准正态分布的查找表
norm_lookuptable = np.zeros(table_size, dtype=np.float)  # -5000，5000，映射到 -5，5
for index in range(table_size):
    int_x = index - table_offset # index=5000 值最高
    float_x = int_x / scale
    norm_lookuptable[index] = norm_pdf.pdf(float_x, 0.0, 1) # 正态分布，越靠近0越大
    pass
print("generate lookuptable")


@jit(nopython=True)
def lookup(raw_index): # raw_index = 0值最高
    index = int(raw_index * scale + table_offset)
    if index > (table_size - 1) or index < 0:
        return 0.0
    else:
        return norm_lookuptable[index]


"""
基于机器人感知到的其他物体的状态、构建机器人感知到的地图
根据一组机器人的状态（位置、速度等）和感知范围，在地图上构建出这组机器人的感知图，即对机器人周围环境的感知情况进行量化。
具体来说，对于每个机器人，代码首先根据其当前的状态信息（位置、速度、半径等）计算出它在接下来的一个时间步长内可能到达的位置和其感知范围，
然后以这个范围为基础，对周围的栅格（地图上的离散位置）进行遍历，计算出每个栅格的感知概率，并将其叠加到感知图上
human_states中包括了除机器人自身状态以外的其他物体的状态
"""
@jit(nopython=True)
def build_all_human_map(human_states, width, height, x_offset, y_offset, resolution, time_step, position_variance,
                        direction_variance):
    global_gmap = np.zeros((width, height), dtype=np.float_)
    for index in range(human_states.shape[0]):
        gmap = np.zeros((width, height), dtype=np.float_)
        px = human_states[index, 0]
        py = human_states[index, 1]
        vx = human_states[index, 2]
        vy = human_states[index, 3]
        radius = human_states[index, 4]
        p_x = px + x_offset
        p_y = py + y_offset
        center_x = round(p_x / resolution)
        center_y = round(p_y / resolution)
        radius_pixel_len = round(radius / resolution)

        # movement in a time step
        delta_x = vx * time_step
        delta_y = vy * time_step
        move_distance = math.hypot(delta_x, delta_y) # 机器人在时间步长内移动的距离
        # move_radius = round((1.0 * time_step) / resolution)

        # calculate boundary of ROI 感知范围（Region of Interest, ROI）
        roi_x_low_pos = p_x - move_distance
        roi_x_high_pos = p_x + move_distance

        roi_x_low_idx = round(roi_x_low_pos / resolution)
        roi_x_high_idx = round(roi_x_high_pos / resolution)

        roi_y_low_pos = p_y - move_distance
        roi_y_high_pos = p_y + move_distance

        roi_y_low_idx = round(roi_y_low_pos / resolution)
        roi_y_high_idx = round(roi_y_high_pos / resolution)

        roi_width = roi_x_high_idx - roi_x_low_idx
        roi_height = roi_y_high_idx - roi_y_low_idx

        heading_angle = atan2(delta_y, delta_x)  # rad obstacle在当前时间步长内移动的方向（弧度表示）
        sum_probability = 0.0

        loop_cnt = 0
        for y_index in range(roi_y_low_idx, roi_y_high_idx):
            for x_index in range(roi_x_low_idx, roi_x_high_idx):
                if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance: # 除去在矩形中但不在圆形的点
                    continue
                if (y_index - center_y) == 0 and (x_index - center_x) == 0:
                    angle = heading_angle # 机器人在当前时间步长内没有移动，angle 的值就等于 heading_angle
                else:
                    angle = atan2(y_index - center_y, x_index - center_x) # y_index是机器人

                # warp to -pi pi
                delta_angle = angle - heading_angle
                if delta_angle <= -math.pi:
                    delta_angle += 2.0 * math.pi
                elif delta_angle >= math.pi:
                    delta_angle = 2.0 * math.pi - delta_angle

                # position_variance 是指机器人位置的方差，也就是机器人位置不确定性的度量
                # x[0] 当前栅格与原来位置在 y 轴方向上的距离的方差
                x = [
                    position_variance * (y_index - center_y) / (roi_height + 0.000001),
                    position_variance * (x_index - center_x) / (roi_width + 0.000001),
                    direction_variance * delta_angle / math.pi
                ] 
                probability = lookup(x[0]) * lookup(x[1]) * lookup(x[2]) 
                # 越靠近center（原位置），lookup值越大，他的概率越小（因为lookup都小于1）
                gmap[y_index, x_index] += probability # 数组gmap的行代表y轴方向，而列代表x轴方向
                # 在大多数计算机图像库和计算机视觉领域中，坐标系的原点通常是左上角，水平方向为x轴，垂直方向为y轴，这也被称为"行优先"。
                # 因此，数组中的第一维通常对应于y轴方向，第二维对应于x轴方向。这种表示方法已经成为一种约定俗成的惯例。
                sum_probability += probability
                loop_cnt += 1

        for y_index in range(roi_y_low_idx, roi_y_high_idx):
            for x_index in range(roi_x_low_idx, roi_x_high_idx):
                if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance:
                    continue
                if sum_probability == 0.0:
                    gmap[y_index, x_index] = 0.0
                else:
                    gmap[y_index, x_index] = gmap[y_index, x_index] / sum_probability
        global_gmap += gmap
    return global_gmap


@jit(nopython=True)
def calculate_map_pixel(map, x_low, x_high, y_low, y_high, resolution, search_radius, center_x_idx, center_y_idx, width,
                        height):
    sum_probability = 0.0
    for index_y in range(y_low, y_high):
        for index_x in range(x_low, x_high):
            if math.hypot((index_x - center_x_idx), (index_y - center_y_idx)) * resolution > search_radius:
                continue
            if (index_x < width) and (index_y < height):
                sum_probability += map[index_y, index_x] # 因为只有存在human时map那个位置点才有值（gmap），否则为0
    return sum_probability


class gridmap(object):
    def __init__(self, xlimit, ylimit, resolution=0.02, time_step=0.6, position_variance=3.0, direction_variance=3.0):
        self.time_step = time_step
        self.x_offset = round(xlimit / 2)
        self.y_offset = round(ylimit / 2)
        self.width = round(xlimit / resolution)
        self.height = round(ylimit / resolution)
        self.gmap = np.zeros((self.width, self.height), dtype=np.float)
        self.human_states = []
        self.human_info_list = []
        self.resolution = resolution

        self.agent_center_x = 0
        self.agent_center_y = 0
        self.check_pixel_radius = 1

        self.position_variance = position_variance
        self.direction_variance = direction_variance
        pass

    def clear_map(self):
        self.gmap = np.zeros((self.width, self.height), dtype=np.float)
        self.human_states = []
    
    # 计算代理(agent)碰撞的概率值
    # 设置了一个搜索半径 = 人类的半径+robot的半径，以robot的位置为中心，搜索半径为半径的矩形一行行搜索
    # 把小于搜索半径的点map[index_y, index_x]都加到概率上来，因为只有人类搜索范围的map才有值，
    # 所以就可以计算成重叠部分的概率了
    def compute_occupied_probability(self, agent_state: FullState):
        agent_x = agent_state.px + self.x_offset
        agent_y = agent_state.py + self.y_offset
        center_x_idx = round(agent_x / self.resolution)
        center_y_idx = round(agent_y / self.resolution)

        self.agent_center_x = center_x_idx
        self.agent_center_y = center_y_idx

        if len(self.human_states) > 0:
            human_radius = self.human_states[0].radius
        else:
            human_radius = 0.0
        search_radius = agent_state.radius + human_radius
        x_low_raw = 0 if (agent_x - search_radius) < 0 else (agent_x - search_radius)
        x_low = round(x_low_raw / self.resolution) # 四舍五入取整
        x_high = round((agent_x + search_radius) / self.resolution)

        y_low_raw = 0 if (agent_y - search_radius) < 0 else (agent_y - search_radius)
        y_low = round(y_low_raw / self.resolution)
        y_high = round((agent_y + search_radius) / self.resolution)

        self.check_pixel_radius = round(search_radius / self.resolution)

        sum_probability = calculate_map_pixel(self.gmap, x_low, x_high, y_low, y_high, self.resolution, search_radius,
                                              center_x_idx, center_y_idx, self.width, self.height)
        return sum_probability

    def get_visualized_map(self):
        visual_map = np.zeros_like(self.gmap, dtype=np.uint8)
        for index_y in range(self.gmap.shape[0]):
            for index_x in range(self.gmap.shape[1]):
                pixel = self.gmap[index_y, index_x] * 255 * 30
                pixel = 255 if pixel > 255 else pixel
                visual_map[index_y, index_x] = 255 - pixel

        visual_map = cv2.circle(visual_map, (self.agent_center_y, self.agent_center_x),
                                self.check_pixel_radius,
                                color=[0, 0, 255],
                                thickness=2)
        return visual_map

    def build_map_cpu(self, human_states, rebuild_map=True):

        self.human_states = human_states
        human_lists = []
        if len(human_lists) != len(human_states):
            pass
        check_list = []
        for human in human_states:
            human_info = [human.px, human.py, human.vx, human.vy, human.radius]
            human_lists.append(human_info)

        if human_lists == self.human_info_list and rebuild_map == False:
            return
        self.human_info_list = human_lists
        human_info_array = np.array(human_lists)
        map = build_all_human_map(human_info_array, self.width, self.height, self.x_offset, self.y_offset,
                                  self.resolution, self.time_step, self.position_variance, self.direction_variance)
        self.gmap = map
        return


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}
    # render.modes 是 OpenAI Gym 环境中 metadata 字典的一个键，用于指定环境支持哪些渲染输出模式。
    # 以下是一些常见的渲染模式：
    # human: 在屏幕上显示环境并交互。此模式下通常会有一个可视化窗口。
    # rgb_array: 返回一张 RGB 像素图像（NumPy 数组）表示的环境。这通常用于渲染环境的图像输出到一个视频文件中。
    # ansi: 返回一个 ASCII 码字符块表示的环境。这通常用于在控制台中显示环境状态，便于调试和测试。


    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        self.map = gridmap(xlimit=12, ylimit=12, resolution=0.02, time_step=0.8)

        self.file_path_prefix = "crowd_nav/map_debug/"
        self.exec_times = 0

    # 在crowd_nav的configs env.config里
    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.stationary_penalty = config.getfloat('reward', 'stationary_penalty')
        self.stationary_penalty_dist = config.getfloat('reward', 'stationary_penalty_dist')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {
                'train': np.iinfo(np.uint32).max - 2000,
                'val': config.getint('env', 'val_size'),
                'test': config.getint('env', 'test_size')
            }
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError

        # reward相关的参数
        self.agent_timestep = config.getfloat("reward", "agent_timestep")
        self.human_timestep = config.getfloat("reward", "human_timestep")
        self.reward_increment = config.getfloat("reward", "reward_increment")
        self.position_variance = config.getfloat("reward", "position_variance")
        self.direction_variance = config.getfloat("reward", "direction_variance")

        self.map = gridmap(xlimit=12,
                           ylimit=12,
                           resolution=0.02,
                           time_step=self.human_timestep,
                           position_variance=self.position_variance,
                           direction_variance=self.direction_variance)

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            # 生成human_num个动态的人，并根据随机概率生成静态的人
            # 生成动态的人
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())

            #生成静态的人
            static_human_num_dict = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            # dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            # static = True if np.random.random() < 0.2 else False
            # static_num = np.random.randint(1, 5) # 生成1 2 3 4 整数
            prob = np.random.random()
            for key, value in sorted(static_human_num_dict.items()):
                if prob - value <= 0:
                    static_human_num = key
                    break
                else:
                    prob -= value


            self.human_num = human_num + static_human_num
          

            # randomly initialize static objects in a square of (width, height)
            width = 4
            height = 8
            if static_human_num == 0:
                human = Human(self.config, 'humans')
                human.set(0, -10, 0, -10, 0, 0, 0)
                self.humans.append(human) # 初始化human并移到场景外

            # 使用循环创建多个人类，并控制它们之间的变量和碰撞检测。每次循环都会创建一个新的Human对象并初始化。
            # 由于可能出现机器人和人类重叠的情况，因此使用while True循环来检测是否存在这种情况。
            for i in range(static_human_num):
                human = Human(self.config, 'humans')
                if np.random.random() > 0.5:
                    sign = -1
                else:
                    sign = 1
                while True:
                    px = np.random.random() * width * 0.5 * sign
                    py = (np.random.random() - 0.5) * height
                    collide = False
                    # 如果机器人和某个人类（human）之间的距离小于它们两者半径之和再加上self.discomfort_dist的距离，则认为它们发生了碰撞，
                    # 程序将设置collide标记为True，然后退出for循环。其中，norm函数是计算两个点之间的欧几里得距离，
                    # px和py是机器人的坐标，agent.px和agent.py是人类（或机器人）的坐标，
                    # human.radius和agent.radius分别表示人类（或机器人）的半径，self.discomfort_dist表示碰撞的最小距离
                    
                    # 将机器人和人类列表 self.humans 合并为同一个列表,机器人与周围的人类都同时被考虑在内，这样能够更方便地遍历整个场景中的所有实体，并进行必要的碰撞检测
                    for agent in [self.robot] + self.humans:
                        if norm(
                            (px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                            collide = True
                            break
                    if not collide:
                        break
                human.set(px, py, px, py, 0, 0, 0)
                self.humans.append(human)
            

        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times): # 当 self.human_times 列表中的元素全部不是 True 时，执行
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())# 目标速度向量
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref) # 重新缩放为一个单位向量
                sim.setAgentPrefVelocity(i, tuple(vel_pref)) # 当前代理的首选速度向量，在仿真中计算代理的下一个位置和方向
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time 
                    # 记录每个人类代理模型完成交互行为的时间，从而让后续的分析或控制可以使用这个信息

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = { # 存储了计数器的偏移量，用于在不同的数据集划分中记录不同的文本案例数量
                'train': self.case_capacity['val'] + self.case_capacity['test'],
                'val': 0,
                'test': self.case_capacity['val']
            }
            
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            self.agent_prev_vx = None
            self.agent_prev_vy = None
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                # test情况
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        '''        
        coordinates坐标并不是固定的传感器类型，它指的是获取物体或者障碍物的坐标信息，通常可以由多种传感器和技术来实现。
        比如，可以通过激光雷达扫描周围环境，获取环境中物体和障碍物的位置信息，也可以通过摄像头实现物体的目标跟踪和坐标获取，
        又或者通过超声波、磁力计、惯性导航等多种传感器结合来获取机器人自身的位置和姿态信息。因此
        coordinates一般不属于任何特定的传感器类型，而是指针对具体情况，机器人采用不同的传感器和算法来获取坐标信息

        '''

        return ob

    def onestep_lookahead(self, action, rebuild=True):
        return self.step(action, update=False, rebuild=rebuild)

    # env.step
    def step(self, action, update=True, rebuild=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible: # 设置成false是为了确保障碍机器人不能主动避开无人机器人
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))  # human的state和他的观测ob这里human用orca policy运行

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx # action是机器人的
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            # point_to_segment_dist返回点和线段之间的最短距离 就得这么算！不能直接位置相减
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # begin = time()
        self.map.build_map_cpu(self.humans, rebuild)
        # if not update:
        #     print("rebuild: ", rebuild, "build map time: ", 81 * (time() - begin))

        agent_fullstate = FullState(self.robot.px + action.vx * self.agent_timestep,
                                    self.robot.py + action.vy * self.agent_timestep, self.robot.vx, self.robot.vy,
                                    self.robot.radius, self.robot.gx, self.robot.gy, self.robot.v_pref,
                                    self.robot.theta)

        # begin = time()
        collision_probability = self.map.compute_occupied_probability(agent_fullstate)
        # if not update:
        #     print("rebuild: ", rebuild, "build map time: ", 81 * (time() - begin))

        stationary_dist = ((self.robot.py + self.circle_radius)**2 + (self.robot.px**2))**(1 / 2)

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx**2 + dy**2)**(1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        # 靠近goal的奖励
        previous_position =np.array(self.robot.px,self.robot.py) 
        process_goal = np.linalg.norm(previous_position - np.array(self.robot.get_goal_position())) - \
            np.linalg.norm(end_position - np.array(self.robot.get_goal_position())) 

        # 平稳移动 move smoothly
        last_mag = np.sqrt(self.robot.vx ** 2 + self.robot.vy ** 2)
        mag = np.sqrt(action.vx ** 2 + action.vy ** 2)
        if last_mag > 0 and mag > 0:
            dot_product = np.dot([self.robot.vx, self.robot.vy], [action.vx, action.vy])
            cos_theta = dot_product / (last_mag * mag)
            smoothness_reward = np.abs(cos_theta) * 0.1
            # 如果 cos_theta 的绝对值越接近 1，则说明机器人的速度方向变化越平滑。
            # 如果 cos_theta 的绝对值越接近 0，则说明机器人的速度方向变化越突然，平滑性越差。


        stationary_state = False
        if self.robot.kinematics == 'holonomic':
            if abs(action.vx) <= 0.0001 and abs(action.vy) <= 0.0001:
                stationary_state = True
        else:
            if abs(action.v) <= 0.0001:
                stationary_state = True

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif stationary_state: # update 静止不动
            reward = self.stationary_penalty
            done = False
            info = Stationary()
        elif process_goal > 0:
            reward =  0.1 # 靠近goal的奖励
            done = False
            info = Process()
        else:
            reward = -0.05 # 远离goal的惩罚
            done = False
            info = Nothing()

        # 安全性分数        
        if dmin < self.discomfort_dist: 
            discomfort_reward = -1 * self.discomfort_dist * self.discomfort_penalty_factor # 小于安全距离的惩罚
            info = Danger(dmin)
        else:
            info = Nothing()
        
        collision_probability_reward = -0.25 * collision_probability * self.reward_increment 
        reward += smoothness_reward + collision_probability_reward + discomfort_reward
        done = False
        

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        # self.agent_prev_vx = action.vx
        # self.agent_prev_vy = action.vy
        if self.robot.kinematics == 'holonomic':
            self.agent_prev_vx = action.vx
            self.agent_prev_vy = action.vy
        else:
            self.agent_prev_vx = action.v * np.cos(action.r + self.robot.theta)
            self.agent_prev_vy = action.v * np.sin(action.r + self.robot.theta)

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [
                        plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                        for i in range(len(self.humans))
                    ]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [
                        plt.text(agents[i].center[0] - x_offset,
                                 agents[i].center[1] - y_offset,
                                 '{:.1f}'.format(global_time),
                                 color='black',
                                 fontsize=14) for i in range(self.human_num + 1)
                    ]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color,
                                               ls='solid')
                    human_directions = [
                        plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                   color=cmap(i),
                                   ls='solid') for i in range(self.human_num)
                    ]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [
                plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))
            ]
            human_numbers = [
                plt.text(humans[i].center[0] - x_offset,
                         humans[i].center[1] - y_offset,
                         str(i),
                         color='black',
                         fontsize=12) for i in range(len(self.humans))
            ]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            # if self.attention_weights is not None:
            #     attention_scores = [
            #         plt.text(-5.5,
            #                  5 - 0.5 * i,
            #                  'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))
            #     ]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta)))
                               for state in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(
                            ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [
                patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                for orientation in orientations
            ]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [
                        patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color, arrowstyle=arrow_style)
                        for orientation in orientations
                    ]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    # if self.attention_weights is not None:
                    #     human.set_color(str(self.attention_weights[frame_num][i]))
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy, agent.vx, agent.vy,
                                                             agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                # z = np.reshape(z, (16, 5))
                z = np.reshape(z, (len(rotations), len(speeds) - 1))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
import time as t


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    # 更新目标模型（target model）复制模型参数实现模型参数的稳定，从而改善模型训练过程
    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # 在给定环境中运行多次机器人的运动策略，
    # 计算成功率、碰撞率和导航时间等指标。并根据传入的参数更新记忆存储器
    # @profile
    def run_k_episodes(self,
                       k,
                       phase,
                       update_memory=False,
                       imitation_learning=False,
                       episode=None,
                       print_failure=False):
        # logging("run episodes: %d"%(k))
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        time_begin = 0

        for i in range(k):
            time_begin = t.time()
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.robot.act(ob) # agent的更新actionXY
                ob, reward, done, info = self.env.step(action) # 执行action动作并更新环境的状态
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger): # info在crowd_sim设置了
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            # updatememory中传入的相当于是轨迹，在updatememory中计算state的value，然后存入 memory buffer 中
            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            # 计算Q* （2） gamma = 0.9
            # 将每个奖励 reward 乘以 gamma^(t*dt*v_pref)，t 表示当前奖励的时间步，dt 表示时间步长，v_pref 表示参考速度。
            # 对于所有的奖励结果进行加权求和，即得到当前动作的加权奖励结果。
            # 其中，gamma^(t*dt*v_pref)可以理解为根据时间步长的增加，对后续奖励的贡献逐渐减小的一种加权方式
            # 而 sum 函数实现了对列表中所有元素的求和操作。
            cumulative_rewards.append(
                sum([
                    pow(self.gamma, t * self.robot.time_step * self.robot.v_pref) * reward
                    for t, reward in enumerate(rewards)
                ]))

            print("i/k: %d/%d, robot pos: %s, %s, time consuming is:%s secs. %s " %
                  (i, k, self.robot.px, self.robot.py, t.time() - time_begin, info.__str__()))

        # hasattr判断object对象中是否存在name属性
        if hasattr(self.robot.policy, "use_rate_statistic"):
            if self.robot.policy.use_rate_statistic:
                statistic_info = self.robot.policy.get_act_statistic_info()
                print("ACT Usage: ", statistic_info) # 20个中，每个局部最优解出现的次数
        # print(self.robot.policy.model.actf_enncoder.act_fn.act_cnt)
        success_rate = success / k
        collision_rate = collision / k

        # 这是一个 Python 中的断言语句（assert statement），用于检查一个条件是否为真。
        # 具体地，该语句用于断言 “success”、“collision” 和 “timeout” 三个变量之和是否等于另一个变量 “k”。
        # 如果该条件为 False，则会抛出一个 AssertionError 异常，可以用 try/except 语句捕获并处理该异常。
        # 如果该条件为 True，则程序会正常执行下去。
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info(
            '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.format(
                phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, average(cumulative_rewards)))
        
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    """
    根据传入的状态、动作和奖励值，更新记忆存储器中的数据。
    其中，在更新 “值函数” 时，采用了不同的方法，具体来说，如果是在进行 “模仿学习”（imitation learning），则采用的是累积折扣奖励值；
    如果不是，则采用 Q-Learning 的方式进行更新。最后将更新后的数据存储到记忆存储器中
    """
    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)  #CADRL.transform(state)
                # 这里actenvcarl.transform是继承的MultiHumanRL

                # update states after i
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([
                    pow(self.gamma,
                        max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward * (1 if t >= i else 0)
                    for t, reward in enumerate(rewards)
                ])
            else:
                # DQN
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    action_value, _ = self.target_model(next_state.unsqueeze(0)) # actenvcarl
                    value = reward + gamma_bar * action_value.data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

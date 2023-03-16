from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        """
        predict 函数的输入是一个名为 state 的对象，其包含代理和其环境的当前状态，如其位置、速度、目标位置以及环境中的其他代理和障碍物的信息。
        然后它返回一个 ActionXY 对象，其中包含代理的更新速度 rvo2
        """
        # print("self.policy.last_state: ",self.policy.last_state)
        return action

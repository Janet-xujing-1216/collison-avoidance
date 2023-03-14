from gym.envs.registration import register

# 将一个自定义的环境注册到 Gym 的环境列表中
register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSim-v1',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSim-v2',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSim-v3',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSim-v4',
    entry_point='crowd_sim.envs:CrowdSim',
)
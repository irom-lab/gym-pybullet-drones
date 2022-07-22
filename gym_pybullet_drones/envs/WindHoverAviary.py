
from gym_pybullet_drones.assets.wind import Wind
from gym_pybullet_drones.envs.single_agent_rl import HoverAviary

class WindHoverAviary(HoverAviary, Wind):
    def __init__(self, **kwargs):
        HoverAviary.__init__(self, **kwargs)
        Wind.__init__(self, **kwargs)

from gym_pybullet_drones.assets.wind import Wind
from .CtrlAviary import CtrlAviary


class WindCtrlAviary(CtrlAviary, Wind):
    def __init__(self, **kwargs):
        CtrlAviary.__init__(self, **kwargs)
        Wind.__init__(self, **kwargs)

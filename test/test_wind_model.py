import time
import numpy as np
from gym_pybullet_drones.envs.residual_rl.HoverResidualAviary import WindHoverResidualAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.residual_rl.BaseResidualAviary import ActionType
from gym_pybullet_drones.utils.Logger import Logger


episode_len_sec = 5
act = ActionType.RES
physics = Physics.PYB
env = WindHoverResidualAviary(
    gui=True,
    drone_model=DroneModel.X500,
    act=act,
    aggregate_phy_steps=4,  # 60 Hz
    episode_len_sec=episode_len_sec,
    physics=Physics.PYB,
    fixed_init_pos=[[0,0,1]],
    # wind
    wind_model='aero_drag',
    wind_vector=np.array([5, 0, 0]),
    wind_num_frame=3, 
    wind_frame_skip=2, 
    max_wind=10,
    wind_obs_freq=120,
)

pb_logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                    num_drones=1)
obs = env.reset()
reward_total = 0
for i in range(int(episode_len_sec * env.SIM_FREQ / env.AGGR_PHY_STEPS)):
    action = np.array([0,0,0,0])  # no residual

    obs, reward, done, info = env.step(action)
    raw_obs = info['raw_obs']
    pb_logger.log(
        drone=0,
        timestamp=i / env.SIM_FREQ,
        state=np.hstack([
            raw_obs[0:3],
            np.zeros(4), raw_obs[3:15],
            np.resize(action, (4))
        ]),
        # control=action,
        control=np.zeros(12),
    )

    print(reward)
    reward_total += reward
    time.sleep(0.05)

env.close()
print('Total reward: ', reward_total)
pb_logger.plot()

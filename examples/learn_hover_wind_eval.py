"""

"""
import os
import time
import gym
import torch
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool, ensure_directory_hard
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


if __name__ == "__main__":
    log_dir = 'logs/test_hover_no_wind_dyn_ppo_v1/'

    n_envs = 1
    wind_model = 'simple'
# drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7"
    wind_force = [0.0,0,0]
    aggregate_phy_steps = 5
    act = ActionType.DYN
    use_normalize = True
    total_timesteps = 500000
    learning_rate = 1e-3
    ent_coef = 1.0 # 'auto
    model_type = log_dir.split('_')[-2]

    env_kwargs = dict(act=act,
                    aggregate_phy_steps=aggregate_phy_steps,
                    physics=Physics.PYB_WIND,   # Drag model in PyBullet, added wind),
                    wind_model=wind_model,
                    wind_force=wind_force,
                    use_normalize=use_normalize,
                    fixed_init=True)

    # #### Check the environment's spaces ########################
    env = make_vec_env(HoverAviary, 
                        env_kwargs=env_kwargs,
                        n_envs=n_envs,
                        seed=0
                        )
    if model_type == 'td3':
        model = TD3.load(log_dir+'best_model')
    elif model_type == 'sac':      
        model = SAC.load(log_dir+'best_model')
    elif model_type == 'ppo':
        model = PPO.load(log_dir+'best_model')

    #### Show (and record a video of) the model's performance ##
    env = HoverAviary(gui=True,
                    record=True,
                    video_path=log_dir+'best_video.mp4',
                    act=act,
                    physics=Physics.PYB_WIND,
                    aggregate_phy_steps=aggregate_phy_steps,
                    wind_model=wind_model,
                    wind_force=wind_force,
                    use_normalize=use_normalize,
                    fixed_init=True,
                    )
    pb_logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                        num_drones=1
                        )
    obs = env.reset()
    start = time.time()
    episode_sec = 5
    reward_total = 0
    for i in range(int(episode_sec*env.SIM_FREQ/env.AGGR_PHY_STEPS)):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        print(action)
        obs, reward, done, info = env.step(action)
        pb_logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=action,
                   )
        reward_total += reward
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()

        time.sleep(0.1)
    env.close()
    pb_logger.plot()
    print('Total reward: ', reward_total)

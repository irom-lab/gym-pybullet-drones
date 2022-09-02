import os
import time
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.residual_rl.HoverResidualAviary import WindHoverResidualAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.residual_rl.BaseResidualAviary import ActionType

import argparse
from omegaconf import OmegaConf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--cfg_file",
                        help="cfg file path",
                        type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    ########################
    act = ActionType.RES
    physics = Physics.PYB
    env_kwargs = dict(
        drone_model=DroneModel.X500,
        act=act,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        physics=physics,
        wind_model=cfg.wind_model,
        wind_force=cfg.wind_force,
        use_normalize=cfg.use_normalize,
        fixed_init_pos=[cfg.fixed_init_val],  # add dimension
        rate_residual_scale=cfg.rate_residual_scale,
        thrust_residual_scale=cfg.thrust_residual_scale
    )

    #### Check the environment's spaces ########################
    env = make_vec_env(WindHoverResidualAviary,
                       env_kwargs=env_kwargs,
                       n_envs=cfg.num_env,
                       seed=0)
    best_model_path = os.path.join(cfg.log_dir , 'best_model')
    if cfg.model_type == 'td3':
        model = TD3.load(best_model_path)
    elif cfg.model_type == 'sac':
        model = SAC.load(best_model_path)
    elif cfg.model_type == 'ppo':
        model = PPO.load(best_model_path)

    #### Show (and record a video of) the model's performance ##
    env = WindHoverResidualAviary(
        gui=True,
        record=True,
        video_path=os.path.join(cfg.log_dir, 'best_video.mp4'),
        act=act,
        physics=physics,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        wind_model=cfg.wind_model,
        wind_force=cfg.wind_force,
        use_normalize=cfg.use_normalize,
        fixed_init_pos=[cfg.fixed_init_val],  # add dimension
        rate_residual_scale=cfg.rate_residual_scale,
        thrust_residual_scale=cfg.thrust_residual_scale
    )
    pb_logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                       num_drones=1)
    obs = env.reset()
    start = time.time()
    reward_total = 0
    for i in range(int(cfg.episode_len_sec * env.SIM_FREQ / env.AGGR_PHY_STEPS)):
        action, _states = model.predict(obs, deterministic=True)

        # action = np.array([0.0, 0.0, 0.0, 0.0])
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
        reward_total += reward
        if i % env.SIM_FREQ == 0:
            env.render()
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
        time.sleep(0.02)
    env.close()
    pb_logger.plot()
    print('Total reward: ', reward_total)

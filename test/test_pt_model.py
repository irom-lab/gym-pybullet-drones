import os
import time
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.residual_rl.HoverResidualAviary import WindHoverResidualAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.residual_rl.BaseResidualAviary import ActionType
from .run_pt_model import MLP, normalize_obs

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
        seed=cfg.seed,
        drone_model=DroneModel.X500,
        act=act,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        physics=physics,
        #
        wind_model=cfg.wind_model,
        wind_profile=cfg.wind_profile,
        wind_profile_args=cfg.wind_profile_args,
        wind_num_frame=cfg.wind_num_frame, 
        wind_frame_skip=cfg.wind_frame_skip, 
        max_wind=cfg.max_wind,
        wind_obs_freq=cfg.wind_obs_freq,
        #
        init_xy_range=cfg.test_init_xy_range,    # test with randomized init too
        init_z_range=cfg.test_init_z_range,
        # fixed_init_pos=[cfg.fixed_init_val],  # add dimension
        rate_residual_scale=cfg.rate_residual_scale,
        thrust_residual_scale=cfg.thrust_residual_scale
    )

    #### Check the environment's spaces ########################
    env = make_vec_env(WindHoverResidualAviary,
                       env_kwargs=env_kwargs,
                       n_envs=1,    # always eval with 1 env
                       seed=cfg.seed)
    best_model_path = os.path.join(cfg.log_dir , 'best_model')
    if cfg.model_type == 'td3':
        model = TD3.load(best_model_path)
    elif cfg.model_type == 'sac':
        model = SAC.load(best_model_path)
    elif cfg.model_type == 'ppo':
        model = PPO.load(best_model_path)

    input_size = 15
    layer_size_list = [input_size] + cfg.net_arch + [4]
    mlp = MLP(dimList=layer_size_list,
              activation_type='relu',)
    policy_path = '/home/allen/gym-pybullet-drones/logs/no_wind_3/best_model/policy.pth'
    var = torch.load(policy_path)
    mlp.load_weight(var)

    #### Show (and record a video of) the model's performance ##
    env = WindHoverResidualAviary(
        seed=cfg.seed,
        gui=True,
        record=False,
        video_path=os.path.join(cfg.log_dir, 'best_video.mp4'),
        act=act,
        physics=physics,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        #
        wind_model=cfg.wind_model,
        wind_profile=cfg.wind_profile,
        wind_profile_args=cfg.wind_profile_args,
        wind_num_frame=cfg.wind_num_frame, 
        wind_frame_skip=cfg.wind_frame_skip, 
        max_wind=cfg.max_wind,
        wind_obs_freq=cfg.wind_obs_freq,
        #
        init_xy_range=cfg.test_init_xy_range,    # test with randomized init too
        init_z_range=cfg.test_init_z_range,
        # fixed_init_pos=[cfg.fixed_init_val],  # add dimension
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
        
        # Inference
        # with torch.no_grad():
        #     action_scaled = mlp.forward(obs)

        # # Convert to numpy, and reshape to the original action shape
        # action_scaled = action_scaled.cpu().numpy().flatten()

        # # Unscale action
        # low, high = -np.ones((4)), np.ones((4))
        # action = low + (0.5 * (action_scaled + 1.0) * (high - low))
        
        # action = np.array([0.0, 0.0, 0.0, 0.0])
        print(action)
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
        time.sleep(0.1)
    env.close()
    pb_logger.plot()
    print('Total reward: ', reward_total)

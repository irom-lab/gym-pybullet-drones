import os
import sys
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.residual_rl.HoverResidualAviary import WindHoverResidualAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.residual_rl.BaseResidualAviary import ActionType
from script.custom_callback import CustomCallback

import wandb
import logging
import argparse
from omegaconf import OmegaConf
# from wandb.integration.sb3 import WandbCallback


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--cfg_file",
                        help="cfg file path",
                        type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    ################### Logging ###################
    log_file = os.path.join(cfg.log_dir, 'log.log')
    log_fh = logging.FileHandler(log_file, mode='w+')
    log_sh = logging.StreamHandler(sys.stdout)
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    # Possible levels: DEBUG, INFO, WARNING, ERROR, CRITICAL    
    logging.basicConfig(format=log_format, level='INFO', 
        handlers=[log_sh, log_fh])

    if cfg.use_wandb:
        wandb.init(entity='allenzren',
                   project=cfg.project,
                   name=cfg.run,
                #    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                )
        wandb.config.update(cfg) 

    # Envs
    act = ActionType.RES
    physics = Physics.PYB
    train_env_kwargs = dict(
        seed=cfg.seed,
        drone_model=DroneModel.X500,
        act=act,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        physics=physics,
        #
        wind_model=cfg.wind_model,
        # wind_force=cfg.wind_force,
        wind_vector=np.array(cfg.wind_vector),
        wind_num_frame=cfg.wind_num_frame, 
        wind_frame_skip=cfg.wind_frame_skip, 
        max_wind=cfg.max_wind,
        wind_obs_freq=cfg.wind_obs_freq,
        #
        init_xy_range=cfg.init_xy_range,    # train with randomized init
        init_z_range=cfg.init_z_range,
        rate_residual_scale=cfg.rate_residual_scale,
        thrust_residual_scale=cfg.thrust_residual_scale)
    eval_env_kwargs = dict(
        seed=cfg.seed,
        drone_model=DroneModel.X500,
        act=act,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        physics=physics,
        #
        wind_model=cfg.wind_model,
        # wind_force=cfg.wind_force,
        wind_vector=np.array(cfg.wind_vector),
        wind_num_frame=cfg.wind_num_frame, 
        wind_frame_skip=cfg.wind_frame_skip, 
        max_wind=cfg.max_wind,
        wind_obs_freq=cfg.wind_obs_freq,
        #
        init_xy_range=cfg.init_xy_range,    # test with randomized init too
        init_z_range=cfg.init_z_range,
        # fixed_init_pos=[cfg.fixed_init_val],  # add dimension
        rate_residual_scale=cfg.rate_residual_scale,
        thrust_residual_scale=cfg.thrust_residual_scale)

    # #### Check the environment's spaces ########################
    env = make_vec_env(WindHoverResidualAviary,
                       env_kwargs=train_env_kwargs,
                       n_envs=cfg.num_env,
                       seed=10)
    logging.info("Action space: {}".format(env.action_space))
    logging.info("Observation space: {}".format(env.observation_space))

    # Separate evaluation env
    eval_env = make_vec_env(
        WindHoverResidualAviary,
        env_kwargs=eval_env_kwargs,
        n_envs=1,
        seed=cfg.seed)
    eval_callback = EvalCallback(
        eval_env,
        # callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=0,
        #                                                  verbose=1),
        verbose=1,
        best_model_save_path=cfg.log_dir,
        n_eval_episodes=cfg.n_eval_episodes,
        log_path=cfg.log_dir,
        eval_freq=cfg.eval_freq,
        deterministic=False,  # Use deterministic actions for evaluation
        render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=list(cfg.net_arch),
                         )

    if cfg.model_type == 'sac':
        model = SAC(
            sacMlpPolicy,
            env,
            device=cfg.device,
            gradient_steps=cfg.gradient_steps,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            learning_rate=cfg.learning_rate,
            train_freq=(cfg.train_freq, "step"),
            policy_kwargs=policy_kwargs,
            ent_coef=cfg.ent_coef,
            verbose=1,
            tensorboard_log=os.path.join(cfg.log_dir, 'tb_log'),
        )
    elif cfg.model_type == 'td3':
        model = TD3(
            td3ddpgMlpPolicy,
            env,
            device=cfg.device,
            gradient_steps=cfg.gradient_steps,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            learning_rate=cfg.learning_rate,
            train_freq=(cfg.train_freq, "step"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=os.path.join(cfg.log_dir, 'tb_log'),
        )
    elif cfg.model_type == 'ppo':
        onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[256, dict(vf=[128], pi=[128])])
        model = PPO(
            a2cppoMlpPolicy,
            env,
            device=cfg.device,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            clip_range=cfg.clip_range,
            policy_kwargs=onpolicy_kwargs,
            tensorboard_log=os.path.join(cfg.log_dir, 'tb_log'),
        )
    train_logger = configure(cfg.log_dir, ["stdout", "log", "csv"])
    model.set_logger(train_logger)
    callback_list = [eval_callback]
    if cfg.use_wandb:
        # callback_list += [WandbCallback(
        #                     gradient_save_freq=2000,
        #                     model_save_path=f"test",
        #                     model_save_freq=2000,
        #                     verbose=2,
        #                     )
        #                  ]
        callback_list += [CustomCallback()]                   
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback_list)
    model.save(os.path.join(cfg.log_dir, 'final_model.zip'))

    #### Log training progression ############################
    with np.load(cfg.log_dir + 'evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            logging.info(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    #### Show (and record a video of) the model's performance ##
    best_model_path = os.path.join(cfg.log_dir , 'best_model')
    if cfg.model_type == 'sac':
        model = SAC.load(best_model_path)
    elif cfg.model_type == 'td3':
        model = TD3.load(best_model_path)
    elif cfg.model_type == 'ppo':
        model = PPO.load(best_model_path)
    env = WindHoverResidualAviary(
        seed=cfg.seed,
        gui=True,
        record=True,
        video_path=os.path.join(cfg.log_dir, 'best_video.mp4'),
        act=act,
        physics=physics,
        aggregate_phy_steps=cfg.aggregate_phy_steps,
        episode_len_sec=cfg.episode_len_sec,
        #
        wind_model=cfg.wind_model,
        # wind_force=cfg.wind_force,
        wind_vector=np.array(cfg.wind_vector),
        wind_num_frame=cfg.wind_num_frame, 
        wind_frame_skip=cfg.wind_frame_skip, 
        max_wind=cfg.max_wind,
        wind_obs_freq=cfg.wind_obs_freq,
        #
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
        obs, reward, done, info = env.step(action)
        pb_logger.log(drone=0,
                      timestamp=i / env.SIM_FREQ,
                      state=np.hstack([
                          obs[0:3],
                          np.zeros(4), obs[3:15],
                          np.resize(action, (4))
                      ]),
                      control=np.zeros(12))
        reward_total += reward
        if i % env.SIM_FREQ == 0:
            env.render()
            logging.info(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    pb_logger.plot()
    logging.info('Total reward: {}'.format(reward_total))

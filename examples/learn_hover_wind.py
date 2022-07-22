"""

"""
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
from gym_pybullet_drones.envs.WindHoverAviary import WindHoverAviary
from gym_pybullet_drones.utils.utils import sync, ensure_directory_hard
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

if __name__ == "__main__":

    #### Save directory ########################################
    log_dir = '/home/nate/Documents/gym_logs/test_hover_wind_dyn_ppo_v6/'
    model_type = log_dir.split('_')[-2]
    ensure_directory_hard(log_dir)

    # TODO: use yaml; dump info
    # Envs
    n_envs = 4
    fixed_init_train = None
    fixed_init_valid = [[0, 0, 0.1]]
    wind_model = 'basic'
    wind_force = [10, 0, 0]
    aggregate_phy_steps = 5
    act = ActionType.DYN  # ONE_D_RPM, RPM
    use_normalize = True
    env_kwargs = dict(
        act=act,
        aggregate_phy_steps=aggregate_phy_steps,
        physics=Physics.PYB,  # PYB_WIND is deprecated
        wind_model=wind_model,
        wind_force=wind_force,
        use_normalize=use_normalize,
        fixed_init_pos=fixed_init_train)

    # Time
    total_timesteps = 50000000
    eval_freq = 10000

    # Train
    learning_rate = 1e-3
    batch_size = 256

    # PPO
    n_steps = 1024  # buffer size=n_steps * n_envs
    clip_range = 0.1

    # Off-policy
    buffer_size = 100000
    gradient_steps = 1
    ent_coef = 5.0

    # #### Check the environment's spaces ########################
    env = make_vec_env(WindHoverAviary,
                       env_kwargs=env_kwargs,
                       n_envs=n_envs,
                       seed=0)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    # Separate evaluation env
    env_kwargs['fixed_init_pos'] = fixed_init_valid
    eval_env = make_vec_env(
        WindHoverAviary,
        env_kwargs=env_kwargs,
        n_envs=2,  # n_eval_episodes=10 default
        seed=0)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0,
                                                     verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False)  # Use deterministic actions for evaluation
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        # net_arch=[512, 512, 256, 128]
        net_arch=[256, 128])
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[256, dict(vf=[128], pi=[128])])

    if model_type == 'sac':
        model = SAC(
            sacMlpPolicy,
            env,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            train_freq=(1, "step"),  # (1, "step")
            policy_kwargs=policy_kwargs,
            ent_coef=ent_coef,
            verbose=1,
        )
    elif model_type == 'td3':
        model = TD3(
            td3ddpgMlpPolicy,
            env,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            train_freq=(1, "step"),  # (1, "step")
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    elif model_type == 'ppo':
        model = PPO(
            a2cppoMlpPolicy,
            env,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_steps=n_steps,
            clip_range=clip_range,
            policy_kwargs=onpolicy_kwargs,
        )
    train_logger = configure(log_dir, ["stdout", "log", "csv"])
    model.set_logger(train_logger)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(log_dir + 'final_model.zip')

    #### Print training progression ############################
    with np.load(log_dir + 'evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    #### Show (and record a video of) the model's performance ##
    if model_type == 'sac':
        model = SAC.load(log_dir + 'best_model')
    elif model_type == 'td3':
        model = TD3.load(log_dir + 'best_model')
    elif model_type == 'ppo':
        model = PPO.load(log_dir + 'best_model')
    env = HoverAviary(
        gui=True,
        record=True,
        video_path=log_dir + 'best_video.mp4',
        act=act,
        physics=Physics.PYB_WIND,
        aggregate_phy_steps=aggregate_phy_steps,
        wind_model=wind_model,
        wind_force=wind_force,
        use_normalize=use_normalize,
    )
    pb_logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                       num_drones=1)
    obs = env.reset()
    start = time.time()
    episode_sec = 5
    reward_total = 0
    for i in range(int(episode_sec * env.SIM_FREQ / env.AGGR_PHY_STEPS)):
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
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    pb_logger.plot()
    print('Total reward: ', reward_total)

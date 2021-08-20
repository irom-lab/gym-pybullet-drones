"""

"""
import time
import gym
import numpy as np

from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool, ensure_directory_hard
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


def f_hover(*args, **kwargs):
	return GymEnvWrapper(HoverAviary(**kwargs))


if __name__ == "__main__":

    # TODO: use yaml; vec; dump info
    wind_model = 'simple'
    wind_force = [0,0,0]
    aggregate_phy_steps = 5
    
    log_dir = 'logs/test_hover_wind/'
    ensure_directory_hard(log_dir)
    prefix = ''
    log_tabular_only = True
    check_running = False # running avg
    reuse_replay_buffer = False

    cuda_idx = 0
    cpu_offset = 0
    num_cpus = 4
    n_steps = 1e5

    num_env_train = 4
    batch_T = 2
    num_env_eval = 4
    num_step_per_eval = 240
    discount = 0.99
    batch_size = 256
    min_steps_learn = 1e3
    replay_size = 1e6
    replay_ratio = 1
    learning_rate = 3e-4
    fixed_alpha = None  # learn
    clip_grad_norm = 1e9
    reward_scale = 1
    pretrain_std = 0.75
    action_squash = 1
    
    num_itr_per_log = 100
    log_interval_steps = batch_T*num_env_train*num_itr_per_log

    sampler = GpuSampler(
        EnvCls=f_hover,
        env_kwargs=[dict(act=ActionType.ONE_D_RPM,
                        physics=Physics.PYB_WIND,
                        wind_model=wind_model,
                        wind_force=wind_force,
                        aggregate_phy_steps=aggregate_phy_steps) for _ in range(num_env_train)],
        batch_T=batch_T,
        batch_B=num_env_train,
        max_decorrelation_steps=0,
        eval_n_envs=num_env_eval,
        eval_env_kwargs=[dict(act=ActionType.ONE_D_RPM,
                            physics=Physics.PYB_WIND,
                            wind_model=wind_model,
                            wind_force=wind_force,
                            aggregate_phy_steps=aggregate_phy_steps) for _ in range(num_env_eval)],
        eval_max_steps=num_step_per_eval*num_env_eval
    )

    algo = SAC(discount=discount,
                batch_size=batch_size,
                min_steps_learn=min_steps_learn,
                replay_size=replay_size,
                replay_ratio=replay_ratio,
                learning_rate=learning_rate,
                fixed_alpha=fixed_alpha,
                clip_grad_norm=clip_grad_norm,
                reward_scale=reward_scale,
                )

    model_kwargs = dict(hidden_sizes=[512, 512, 256, 128])
    q_model_kwargs = dict(hidden_sizes=[512, 512, 256, 128])
    agent = SacAgent(model_kwargs=model_kwargs,    # DEFAULT: [256, 256]
                    q_model_kwargs=q_model_kwargs,
                    pretrain_std=pretrain_std,
                    action_squash=action_squash)

    affinity = dict(cuda_idx=cuda_idx,
                    workers_cpus=list(range(cpu_offset, num_cpus+cpu_offset)))
    runner = MinibatchRlEval(algo=algo,
                            agent=agent,
                            sampler=sampler,
                            n_steps=n_steps,
                            log_interval_steps=log_interval_steps,
                            affinity=affinity)

    run_ID = 0
    with logger_context(log_dir, run_ID, prefix, {}, override_prefix=True, log_tabular_only=log_tabular_only, snapshot_mode='last'):
        returns = runner.train()
 
    #### Show (and record a video of) the model's performance ##
    # env = HoverAviary(gui=True,
    #                 record=False,
    #                 physics=Physics.PYB_WIND,
    #                 wind_model=wind_model,
    #                 wind_force=wind_force,
    #                 )
    # logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
    #                 num_drones=1
    #                 )
    # obs = env.reset()
    # start = time.time()
    # for i in range(3*env.SIM_FREQ):
    #     action, _states = model.predict(obs,
    #                                     deterministic=True
    #                                     )
    #     obs, reward, done, info = env.step(action)
    #     logger.log(drone=0,
    #                timestamp=i/env.SIM_FREQ,
    #                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
    #                control=np.zeros(12)
    #                )
    #     if i%env.SIM_FREQ == 0:
    #         env.render()
    #         print(done)
    #     sync(i, start, env.TIMESTEP)
    #     if done:
    #         obs = env.reset()
    # env.close()
    # logger.plot()

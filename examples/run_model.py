"""Script demonstrating the joint use of simulation and control.

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (R, 0).

"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.WindCtrlAviary import WindCtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.PX4Control import PX4Control
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser()
    ARGS = parser.parse_args()

    #
    ARGS.drone = DroneModel.X500  # CF2X
    ARGS.physics = Physics.PYB
    ARGS.obstacles = False

    ARGS.aggregate = True
    ARGS.simulation_freq_hz = 240
    ARGS.control_freq_hz = 48
    ARGS.duration_sec = 6

    ARGS.gui = True
    ARGS.record_video = False
    ARGS.plot = True
    ARGS.user_debug_gui = False

    wind_model = 'basic'
    wind_force = [1000, 0, 0]

    #### Initialize the simulation #############################
    H = 1.0
    H_STEP = 0.0
    R = .3
    INIT_XYZS = np.array([[0, 0, H]])
    INIT_RPYS = np.array([[0, 0, 0]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz /
                         ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    # for i in range(NUM_WP):  # around [R, 0]
    #     # TARGET_POS[i, :] = R * np.cos(
    #     #     (i / NUM_WP) *
    #     #     (2 * np.pi) + np.pi) + INIT_XYZS[0, 0] + R, R * np.sin(
    #     #         (i / NUM_WP) * (2 * np.pi) + np.pi) + INIT_XYZS[0, 1], 0
    #     # TARGET_POS[i, :] = 0.0, 0.0, H
    #     TARGET_POS[i, :] = 0.005 * i, 0.005 * i, H
    fig = plt.figure()
    plt.scatter(TARGET_POS[:, 0], TARGET_POS[:, 1])
    plt.title('Target Position: Point')
    plt.show()
    wp_counters = np.array([0])

    #### Create the environment with or without video capture ##
    # env = CtrlAviary(
    env = WindCtrlAviary(
        drone_model=ARGS.drone,
        num_drones=1,
        fixed_init_pos=INIT_XYZS,
        # initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=ARGS.physics,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=ARGS.record_video,
        obstacles=ARGS.obstacles,
        user_debug_gui=ARGS.user_debug_gui,
        # wind
        wind_model=wind_model,
        wind_force=wind_force
        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz /
                                        AGGR_PHY_STEPS),
                    num_drones=1)

    #### Initialize the controllers ############################
    # ctrl = [DSLPIDControl(drone_model=ARGS.drone)]
    ctrl = [
        PX4Control(drone_model=ARGS.drone, Ts=AGGR_PHY_STEPS / env.SIM_FREQ)
    ]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ /
                                      ARGS.control_freq_hz))  # 240/48=5
    action = {str(0): np.array([0, 0, 0, 0])}
    obs = env.reset()
    START = time.time()
    for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:
            offset = obs[str(0)]["state"][:3] - np.hstack(
                [TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2]])
            print('Offset: ', offset)

            #### Compute control for the current way point #############
            action[str(0)], _, _ = ctrl[0].computeControlFromState(
                # control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                state=obs[str(0)]["state"],
                target_pos=np.hstack(
                    [TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2]]),
                # target_rpy=INIT_RPYS[0, :]
            )
            # print(action[str(0)])

            #### Go to the next way point and loop #####################
            wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP -
                                                                     1) else 0

            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)

        time.sleep(0.1)

        #### Log the simulation ####################################
        print('logger = ' + str(wind_force))
        logger.log(drone=0,
                   timestamp=i / env.SIM_FREQ,
                   state=obs[str(0)]["state"],
                   control=np.hstack([
                       TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2],
                       INIT_RPYS[0, :],
                       np.zeros(6)
                   ]),
                   wind_force=wind_force
                   )

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid")  # Optional CSV save

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()

"""Script demonstrating the joint use of simulation and control.

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser()
    ARGS = parser.parse_args()

    #
    ARGS.drone = DroneModel.CF2X
    ARGS.physics = Physics.PYB
    ARGS.obstacles = False

    ARGS.aggregate = True
    ARGS.simulation_freq_hz = 240
    ARGS.control_freq_hz = 48
    ARGS.duration_sec = 12

    ARGS.gui = True
    ARGS.record_video = False
    ARGS.plot = True
    ARGS.user_debug_gui = False

    #### Initialize the simulation #############################
    H = 1.0
    H_STEP = 0.0
    R = .3
    INIT_XYZS = np.array(
        [[R * np.cos(np.pi / 2), R * np.sin(np.pi / 2) - R, H]])
    INIT_RPYS = np.array([[0, 0, 0]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz /
                         ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R * np.cos(
            (i / NUM_WP) *
            (2 * np.pi) + np.pi / 2) + INIT_XYZS[0, 0], R * np.sin(
                (i / NUM_WP) *
                (2 * np.pi) + np.pi / 2) - R + INIT_XYZS[0, 1], 0
    wp_counters = np.array([0])

    #### Create the environment with or without video capture ##
    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=ARGS.physics,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=ARGS.obstacles,
                     user_debug_gui=ARGS.user_debug_gui)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz /
                                        AGGR_PHY_STEPS),
                    num_drones=1)

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=ARGS.drone)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.control_freq_hz))
    action = {str(0): np.array([0, 0, 0, 0])}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action[str(0)], _, _ = ctrl[0].computeControlFromState(
                control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                state=obs[str(0)]["state"],
                target_pos=np.hstack(
                    [TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2]]),
                target_rpy=INIT_RPYS[0, :])

            #### Go to the next way point and loop #####################
            wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP -
                                                                     1) else 0

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i / env.SIM_FREQ,
                   state=obs[str(0)]["state"],
                   control=np.hstack([
                       TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2],
                       INIT_RPYS[0, :],
                       np.zeros(6)
                   ]))

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

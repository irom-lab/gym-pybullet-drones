import os
from enum import Enum
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.control.PX4Control import PX4Control


class ActionType(Enum):
    """Action type enumeration class."""
    RES = "res"  # Residual to


################################################################################


class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematic information (pose, linear and angular velocities)


################################################################################


class BaseResidualAviary(BaseAviary):
    """Base single drone environment class for reinforcement learning."""

    ################################################################################

    def __init__(
        self,
        seed=42,
        drone_model: DroneModel = DroneModel.X500,
        initial_xyzs=None,
        initial_rpys=None,
        fixed_init_pos=None,
        init_xy_range=[0, 0],
        init_z_range=[1, 1],
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        episode_len_sec: int = 5,
        gui=False,
        record=False,
        video_path=None,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RES,
        # residual
        rate_residual_scale=0.1,
        thrust_residual_scale=1.0,
    ):
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1; `vision_attributes`
        and `dynamics_attributes` are selected based on the choice of `obs`
        and `act`; `obstacles` is set to True and overridden with landmarks for
        vision applications; `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = episode_len_sec
        Ts = aggregate_phy_steps / freq
        self.rate_residual_scale = rate_residual_scale
        self.thrust_residual_scale = thrust_residual_scale

        #### Create integrated controllers #########################
        self.ctrl = PX4Control(drone_model=drone_model, Ts=Ts)
        super().__init__(
            seed=seed,
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            fixed_init_pos=fixed_init_pos,
            init_xy_range=init_xy_range,
            init_z_range=init_z_range,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            video_path=video_path,
            obstacles=False,
            user_debug_gui=
            False,  # Remove of RPM sliders from all single agent learning aviaries
            vision_attributes=False,
            dynamics_attributes=False,
            )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE == ActionType.RES:
            size = 4
        else:
            raise "[ERROR] in BaseSingleAgentAviary._actionSpace()"
        return spaces.Box(low=-1 * np.ones(size),
                          high=np.ones(size),
                          dtype=np.float32)

    ################################################################################

    def _preprocessAction(self, action, verbose=False):
        """
        Use residual. Use current yaw as yaw setpoint.
        """
        rate_residual = action[:-1] * self.rate_residual_scale
        thrust_residual = action[-1] * self.thrust_residual_scale
        if verbose:
            print('Residual: ', rate_residual, thrust_residual)
        self.residual = np.hstack((rate_residual, thrust_residual))

        state = self._getDroneStateVector(0)
        # current_yaw = state[9]
        action, _, _, self.raw_control = self.ctrl.computeControlFromState(
            state=state,
            target_pos=self.TARGET_POS,
            rate_residual=rate_residual,
            thrust_residual=thrust_residual,
            # target_rpy=np.array([0, 0, current_yaw]),
        )
        
        # TODO: use minRPM?
        clipped_action = np.clip(np.array(action), 0, self.MAX_RPM)
        return clipped_action

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            # OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X Y Z | Q1 Q2 Q3 Q4 | R P Y | VX VY VZ | WX WY WZ |  (no RPMs)
            obs_lower_bound = np.array([
                -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
            ])
            obs_upper_bound = np.array([
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ])
            return spaces.Box(low=obs_lower_bound,
                              high=obs_upper_bound,
                              dtype=np.float32)

            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            # OBS OF SIZE 16 (WITH QUATERNION AND no RPMS)
            return obs
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError

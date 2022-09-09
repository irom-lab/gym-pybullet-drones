import numpy as np
from collections import deque
from gym import spaces

from gym_pybullet_drones.assets.wind import Wind
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.residual_rl.BaseResidualAviary import ActionType, ObservationType, BaseResidualAviary
from gym_pybullet_drones.utils.utils import get_frames


class HoverResidualAviary(BaseResidualAviary):
    """Single agent RL problem: hover at position. Using residual model"""

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
        target_pos=[0, 0, 1],
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
        **kwargs,
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(seed=seed,
                         drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         fixed_init_pos=fixed_init_pos,
                         init_xy_range=init_xy_range,
                         init_z_range=init_z_range,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         episode_len_sec=episode_len_sec,
                         gui=gui,
                         record=record,
                         video_path=video_path,
                         obs=obs,
                         act=act,
                         rate_residual_scale=rate_residual_scale,
                         thrust_residual_scale=thrust_residual_scale)
        self.TARGET_POS = target_pos
        self.max_dist = 1

    ################################################################################
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # pos-3; quat-4; rpy-3; vel-3; ang-vel-4; last_clipped_action-4
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(np.array(self.TARGET_POS) - state[0:3])
        dist_ratio = dist / self.max_dist
        reward = max(0, 1-dist_ratio)*0.1
        return reward


    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            self.ctrl.reset()
            return True
        elif state[0] > 1.0 or state[1] > 1.0:
            self.ctrl.reset()
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        #? These values are large
        # MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC  # 3*5=15
        # MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC  # 1*5=5
        MAX_XY = 1
        MAX_Z = 2

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state, clipped_pos_xy,
                                               clipped_pos_z, clipped_rp,
                                               clipped_vel_xy, clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16] / np.linalg.norm(
            state[13:16]) if np.linalg.norm(
                state[13:16]) != 0 else state[13:16]

        # MAX_LIN_VEL_XY = 0.5
        # MAX_LIN_VEL_Z = 0.5
        # MAX_XY = 0.5
        # MAX_Z = 1.5
        # MAX_PITCH_ROLL = np.pi/60
        # MAX_ANG_VEL = 0.5
        
        # clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        # clipped_pos_z = np.clip(state[2], 0.5, MAX_Z)
        # clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        # clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        # clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        # clipped_vel_ang_vel = np.clip(state[13:16], -MAX_ANG_VEL, MAX_ANG_VEL)

        # if self.GUI:
        #     self._clipAndNormalizeStateWarning(state, clipped_pos_xy,
        #                                         clipped_pos_z, clipped_rp,
        #                                        clipped_vel_xy, clipped_vel_z)

        # normalized_pos_xy = state[0:2] / MAX_XY
        # normalized_pos_z = (clipped_pos_z - 0.5) / (MAX_Z - 0.5)
        # normalized_rp = clipped_rp / MAX_PITCH_ROLL
        # normalized_y = state[9] / np.pi  # No reason to clip
        # normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        # normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        # normalized_ang_vel = clipped_vel_ang_vel / MAX_ANG_VEL

        norm_and_clipped = np.hstack([
            normalized_pos_xy, 
            normalized_pos_z, 
            # state[3:7], 
            normalized_rp,
            normalized_y, 
            normalized_vel_xy, 
            normalized_vel_z,
            normalized_ang_vel
        ]).reshape(12, )  # no rpms

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it", self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]"
                .format(state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it", self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]"
                .format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it", self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]"
                .format(state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it", self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]"
                .format(state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it", self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]"
                .format(state[12]))


class WindHoverResidualAviary(HoverResidualAviary, Wind):
    def __init__(self, wind_num_frame=1, 
                       wind_frame_skip=0, 
                       max_wind=10,
                       wind_obs_freq=120,
                       **kwargs):
        # For normalization as state input
        self.max_wind = max_wind

        # Wind history config
        self.wind_num_frame = wind_num_frame
        self.wind_frame_skip = wind_frame_skip        
        self.wind_frame_cover = (
            self.wind_num_frame - 1
        ) * self.wind_frame_skip + self.wind_num_frame  # maxlen in deque; if fewer, randomly sample

        # Parents
        HoverResidualAviary.__init__(self, **kwargs)
        Wind.__init__(self, **kwargs)
        
        # Wind observation frequency
        assert self.SIM_FREQ%wind_obs_freq == 0   # matching pybullet
        self.wind_obs_freq = wind_obs_freq
        self.wind_obs_aggregate_phy_steps = int(self.SIM_FREQ/wind_obs_freq)  


    def reset(self):

        # Reset wind history
        self.wind_frames = deque(maxlen=self.wind_frame_cover)
        self.wind_frames.appendleft(self.rng.random(3)*0.1) # small random value initially
        # self.wind_frames.appendleft(self.wind_vector)

        # Get new obs
        return super().reset()

    
    def _observationSpace(self):
        obs_lower_bound = np.array([
            -1, -1, 0, 
            # -1, -1, -1, -1, 
            -1, -1, -1, 
            -1, -1, -1, 
            -1, -1, -1, 
        ])  # 16
        obs_upper_bound = np.array([
            1, 1, 1, 
            # 1, 1, 1, 1, 
            1, 1, 1, 
            1, 1, 1, 
            1, 1, 1,
        ])
        # Add wind frames - use 3-dim wind vector - normalized
        obs_lower_bound = np.hstack((obs_lower_bound, -np.ones(3*self.wind_num_frame)))
        obs_upper_bound = np.hstack((obs_upper_bound, np.ones(3*self.wind_num_frame)))
        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float32)

    
    def _computeObs(self):
        """Only stacking wind observation. No history for drone state (velocity included). No action."""
        
        # Get state - already normalized
        obs = super()._computeObs()

        # Add wind - normalized
        wind_obs = get_frames(self.wind_frames, 
                              self.wind_num_frame, 
                              self.wind_frame_skip)
        wind_obs = np.clip(wind_obs/self.max_wind, -1, 1)
        # print(wind_obs)

        return np.concatenate((obs, wind_obs))

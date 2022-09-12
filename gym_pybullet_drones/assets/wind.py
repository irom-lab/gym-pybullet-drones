import numpy as np
import pybullet as p
import logging


class Wind():
    """Class for all wind models. Relying on the child also inheriting BaseAviary class for some class attributes used here.
    """

    ################################################################################

    def __init__(
        self,
        wind_model='aero_drag',
        wind_model_args={},
        # wind_force=[0, 0, 0], #used in _drag function (to be depreciated)
        # wind_vector=np.array([1, 0, 0]), #used in _wind_aero_... functions,
        wind_profile='const',
        wind_profile_args={},
        **kwargs,
    ):
        self.wind_model = wind_model
        # self.wind_force = np.array(wind_force)
        # self.wind_vector = wind_vector
        self.wind_profile = wind_profile
        self.wind_profile_args = wind_profile_args
        
        # Constants for WIND_AERO_DRAG
        self.rho = 1.225    # Density of air [kg/m^3]
        # Constants for BLADE_FLAPPING
        self.lambda_0 = 0.075    # Average inflow ratio
        self.theta_0 = np.radians(16)   # Root angle of attack [rad]
        self.thtw = np.radians(-6.6)    # Blade twist [rad]
        self.k_beta = 3 # Hinge spring constant [N.m/rad]
        self.nu_beta = 1.9  # Blade scaled natural frequency (listed as 1.9 on p.4 of 2016, 1.5 in 2020
        self.gamma = 1.04   # Lock number
        # Vehicle Constants
        self.Af = wind_model_args['area']  # Quadrotor frontal area [m^2]
        self.Cd = 0.8   # Quadrotor drag coefficient
        self.Nr = 4 # Number of rotors
        self.Nb = 2 # Number of blades
        self.c = 0.015 # Chord length m
        self.Cla = 2*np.pi # Airfoil lift slope
        self.alpha_ind = np.arctan(self.lambda_0/0.75) # approximation, induced angle of attack [rad]
        self.alpha_eff = self.theta_0 + (3/4)*self.thtw - self.alpha_ind #effective angle of attack [rad]
        self.rbar = 0.0635 # rotor blade length [m] ##NOT SURE IF ROTOR BLADE LENGTH = ROTOR RADIUS !!


    def apply_wind(self, **kwargs):
        if self.wind_model == 'basic':
            self._wind_basic(**kwargs)
        elif self.wind_model == 'aero_drag':
            self._wind_aero_drag(**kwargs)
        elif self.wind_model == 'bf_drag':
            self._wind_bf_drag(**kwargs)
        elif self.wind_model == 'aero_bf_drag':
            self._wind_aero_drag(**kwargs)
            self._wind_bf_drag(**kwargs)
        else:
            raise "Unknown wind model!"
    
    
    def reset_wind_profile(self):
        # Call when reset env

        if self.wind_profile == 'const':
            self.wind_vector = np.array(self.wind_profile_args['wind_vector'])
            # TODO: random

        elif self.wind_profile == 'step_rising':
            # uniform distribution
            t_bottom_low = self.wind_profile_args['t_bottom_low']
            t_bottom_high = self.wind_profile_args['t_bottom_high']
            self.t_bottom = self.rng.random() * (t_bottom_high - t_bottom_low) + t_bottom_low
            
            slope_low = self.wind_profile_args['slope_low']
            slope_high = self.wind_profile_args['slope_high']
            self.slope = self.rng.random() * (slope_high - slope_low) + slope_low

            vel_bottom_low = self.wind_profile_args['vel_bottom_low']
            vel_bottom_high = self.wind_profile_args['vel_bottom_high']
            self.vel_bottom = self.rng.random() * (vel_bottom_high - vel_bottom_low) + vel_bottom_low

            vel_top_low = self.wind_profile_args['vel_top_low']
            vel_top_high = self.wind_profile_args['vel_top_high']
            self.vel_top = self.rng.random() * (vel_top_high - vel_top_low) + vel_top_low

            rise_time = (self.vel_top - self.vel_bottom) / self.slope
            self.t_top = self.t_bottom + rise_time            
            assert self.t_top < self.EPISODE_LEN_SEC

            # noise
            self.vel_std_bottom = self.wind_profile_args['vel_std_bottom']
            self.vel_std_slope = self.wind_profile_args['vel_std_slope']
            self.vel_std_top = self.wind_profile_args['vel_std_top']
            # self.sensor_filter_ratio = self.wind_profile_args['sensor_filter_ratio']
            # self.sensor_std = self.wind_profile_args['sensor_std']

            # dip
            self.vel_std_dip = self.wind_profile_args['vel_std_dip']
            dip_period_low = self.wind_profile_args['dip_period_low']
            dip_period_high = self.wind_profile_args['dip_period_high']
            vel_dip_low = self.wind_profile_args['vel_dip_low']
            vel_dip_high = self.wind_profile_args['vel_dip_high']
            self.dip_period = self.rng.random() * (dip_period_high - dip_period_low) + dip_period_low
            self.t_dip_start = self.rng.uniform(self.t_top, self.EPISODE_LEN_SEC)
            self.t_dip_end = self.rng.uniform(self.t_dip_start, self.t_dip_start+self.dip_period)

            self.vel_dip = self.rng.random() * (vel_dip_high - vel_dip_low) + vel_dip_low
            
            self.vel_past = []
            
            self.sensor_rolling_step = self.wind_profile_args['sensor_rolling_step']
        else:
            raise 'Unknown wind profile!'

    
    def update_wind(self, t):
        """Both simulated wind and simulated sensor measurement. Use smaller noise for sensor measurement; requires more filtering in real."""
        if self.wind_profile == 'const':
            # no need to change wind_vector
            pass
        elif self.wind_profile == 'step_rising':
            # assume in x direction
            if t < self.t_bottom:
                true_noise = -self.rng.gumbel(0, self.vel_std_bottom)
                vel = self.vel_bottom
            elif t < self.t_top:
                true_noise = -self.rng.gumbel(0, self.vel_std_slope)
                vel = (t-self.t_bottom)*self.slope + self.vel_bottom
            elif t < self.t_dip_end and t > self.t_dip_start:
                true_noise = -self.rng.gumbel(0, self.vel_std_dip)
                vel = self.vel_dip + true_noise
            else:
                true_noise = -self.rng.gumbel(0, self.vel_std_top)
                vel = self.vel_top
            vel = max(0, vel + true_noise)
            self.wind_vector = np.array([vel, 0, 0])
            self.vel_past += [vel]

            # Sensor noise proportional to wind noise plus another small noise
            # sensor_noise = true_noise * self.sensor_filter_ratio + self.rng.normal(0, self.sensor_std)
            # sensor = vel + sensor_noise
            self.sensor_vector = np.array([max(self.vel_past[-self.sensor_rolling_step:]), 0, 0])
        else:
            raise 'Unknown widn profile!'
    

    ################################################################################
    def _wind_frame(self, V_infty: np.ndarray) -> np.ndarray:
        """
        This function returns the rotation matrix between wind frame (u1-u2-b3) and body frame.
        That is, the columns of WIND_FRAME are U1, U2, U3.
        Note, U1 is the projection if V_infty onto the rotor hub plane (b1xb2)
        U2 = U1 x U3
        U3 = b3
        
        param@ V_infty: np.ndarray of shape (3,1), wind velocity vector [m/s] in BODY_FRAME coordinates
        return@ WIND_FRAME: np.ndarray of shape (3,3), columns of which are U1, U2, U3
        """
        U3 = np.array([0, 0, 1]).reshape((3,1)) # U3 = e3 in body coordinates
        U1 = np.zeros([3,1])
        U1[0:2] = V_infty[0:2] # Extract horizontal components of V_infty (ignore V_infty[2])
        if np.linalg.norm(U1) != 0:
            U1 = U1 / np.linalg.norm(U1) #Normalize U1
        U2 = -np.cross(U1,U3, axis = 0) #Determine U2 from cross product
        
        return np.hstack((U1, np.hstack((U2,U3)))) #Horizontal concatenation


    ################################################################################

    def _wind_basic(self, 
                    rpm, 
                    nth_drone,
                    wind_force=None,
                    **kwargs):
        """Very basic wind model

        Parameters
        ----------
        wind : ndarray
            (3)-shaped array of floats containing the wind components in x,y,z, in the global frame
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if not wind_force: wind_force = self.wind_force
        base_rot = np.array(
            p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(
                3, 3)
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(
            base_rot,
            drag_factors *
            np.array(self.vel[nth_drone, :] + wind_force)
        )  # vel and wind in global frame, drag in local frame
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                4,
                                forceObj=drag,
                                posObj=[0, 0, 0],
                                flags=p.LINK_FRAME,
                                physicsClientId=self.CLIENT)
    

    ################################################################################

    def _wind_aero_drag(self,
                        rpm,
                        nth_drone,
                        wind_vector_inertial=None,
                        **kwargs):
        """PyBullet implementation of aerodynamic drag forces due to wind.
        from: Craig, William, Derrick Yeo, and Derek A. Paley. "Geometric attitude and position control of a quadrotor in wind." Journal of Guidance, Control, and Dynamics 43.5 (2020): 870-883.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        wind_vector_inertial : ndarray
            (3,1)-shaped array of floats describing the wind vector in the inertial frame
        """
        if not wind_vector_inertial: 
            wind_vector_inertial = self.wind_vector.reshape((3,1))
        
        # Operational Variables
        #omega_j = 8000 #Propeller nominal angular velocity [RPM]
        omega_j = np.mean(rpm) #    Average propeller angular velocity [RPM]
        # NCHECK: Eventually, convert to use individual motor rpms

        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #print("base_rot:")
        #print(base_rot)
        wind_vector = np.matmul(np.linalg.inv(base_rot), wind_vector_inertial) # Body Frame
        #print("wind_vector: [INERTIAL_FRAME]")
        #print(wind_vector_inertial)
        #print("wind_vector: [BODY_FRAME]")
        #print(wind_vector)

        #### Bluff Body Drag
        f_bluff = 0.5*self.rho*np.linalg.norm(wind_vector)*self.Af*self.Cd*wind_vector
        # print("f_bluff: [BODY_FRAME]")
        # print(f_bluff)

        #### Induced Drag
        # rotate wind_vector into BODY_FRAME
        # project wind_vector onto b1xb2 plane
        # (in Paley, V_infty \cdot U1 = wind vector projection)

        # Extract wind frame axis
        U1 = self._wind_frame(wind_vector)[:,0].reshape((3,1))
        f_ind = 0.0

        # Calculate induced drag for each propeller
        for i in range(4):
            omega_j = rpm[i]
            #print('RPM')
            #print(omega_j)
            #don't multiply by self.Nr (number of rotors)
            f_ind += self.Nb/4*self.rho*self.c*self.Cla*self.alpha_eff*np.sin(self.alpha_ind)*omega_j*self.rbar**2*(np.dot(wind_vector.reshape((3,)),U1.reshape((3,))))*U1
        # print("f_ind: [BODY_FRAME]")
        # print(f_ind)

        #### Total Aerodynamic Drag
        f_aero = f_bluff + f_ind
        # print("f_aero: [BODY_FRAME]")
        # print(f_aero)
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=f_aero,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )


    def _wind_bf_drag(self,
                    rpm,
                    nth_drone,
                    wind_vector_inertial=None,
                    **kwargs):
        """PyBullet implementation of blade flapping moment forces due to wind.
        from: Craig, William, Derrick Yeo, and Derek A. Paley. "Geometric attitude and position control of a quadrotor in wind." Journal of Guidance, Control, and Dynamics 43.5 (2020): 870-883.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        wind_vector_inertial : ndarray
            (3,1)-shaped array of floats describing the wind vector in the [inertial frame]
        """
        if not wind_vector_inertial: 
            wind_vector_inertial = self.wind_vector.reshape((3,1))

        # Operational Variables
        omega_j = 8000 #Propeller nominal angular velocity [RPM]
        v_tip = omega_j*2*np.pi/60*self.rbar #Propeller tip speed velocity [m/s]
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        wind_vector = np.matmul(np.linalg.inv(base_rot),wind_vector_inertial) #Body Frame
        #### Determine Wind Frame Axes [NCHECK: Fix This]
        U1 = self._wind_frame(wind_vector)[:,0].reshape((3,1))
        U2 = self._wind_frame(wind_vector)[:,1].reshape((3,1))
        #### Calculate Blade Flapping Variables
        mu = np.linalg.norm(wind_vector)/v_tip #Advance ratio
        chi = np.arctan(mu/self.lambda_0) #
        k_lambda_x = 15*np.pi/23*np.tan(chi/2) #Glauert longitudinal inflow gradient (also k_x)

        beta_1c = -self.gamma/(8*(self.nu_beta**2-1))*self.lambda_0*k_lambda_x
        beta_1s = mu*self.gamma/(4*(self.nu_beta**2-1))*(4/3*self.theta_0+self.thtw-self.lambda_0)
            
        beta_max = np.sqrt(beta_1c**2 + beta_1s**2)
        # NCHECK: Is there a better way than to apply this conditional?
        if np.linalg.norm(wind_vector) == 0:
            phi_d = 0
        else:
            phi_d = np.arctan(beta_1s/beta_1c)-np.pi/2
        #### Calculate Single Hub Moment
        M_single_hub = self.Nb/2*self.k_beta*beta_max*(np.cos(phi_d)*U1+np.sin(phi_d)*U2)
        #### Calculate Total Hub Moment
        M_aero = np.array([[self.Nr*self.k_beta*beta_max*np.sin(phi_d)*np.dot(U2.reshape(3,),np.array([1,0,0]))],[self.Nr*self.k_beta*beta_max*np.sin(phi_d)*np.dot(U2.reshape(3,),np.array([0,1,0]))],[0]])
        # NCHECK: Need to check frames
        print('M_AERO: [BODY_FRAME]')
        print(M_aero)
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=M_aero,
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )


    @staticmethod
    def wind_function(time: float, gust_peak_time: float, gust_type: int) -> float:
        """
        This function returns a scalar gust value depending on the time and gust type.
        Based on: Cole, Kenan, and Adam Wickenheiser. "Spatio-Temporal Wind Modeling for UAV Simulations." arXiv e-prints (2019): arXiv-1905.
        
        param@ time: current time [s]
        param@ gust_peak_time: time of gust peak [s]
        param@ gust_type: integer to select gust options.
            gust_type == 0 -> semi-square impulse
            gust_type == 1 -> custom (change values in the function as you see fit)
            ...            -> add additional types as desired

        return@ float representing a 'gust scalar', to be added to a nominal wind [m/s]
        """
        if gust_type == 0: # Semi-Square Pulse
            tg1 = 0.1 #rising time
            tg2 = 0.1 #falling time
            th = 1  #holding time
            g1 = 1  #peak gust value
            g3 = 6 #change 'curvature'
            g4 = .1 #change magnitude of dip before gust
            g5 = .1 #change magnitude of dip after gust
        elif gust_type == 1: # Custom
            tg1 = 1 #rising time
            tg2 = 1 #falling time
            th = 1  #holding time
            g1 = 1  #peak gust value
            g3 = .3 #change 'curvature' of rising and falling sections
            g4 = 1 #change magnitude of dip before gust
            g5 = 2 #change magnitude of dip after gust
        else:
            logging.error('No known gust_type specified.')
            return 0

        g2r = 2*g3/tg1
        g2f = 2*g3/tg2
        
        t = time - gust_peak_time
        
        if t<0.5*tg1: #Rising portion
            return g1*(1-(g2r*t-g3)**2)*np.exp(-(g2r*t-g3)**2/g4)
        elif t<0.5*tg1+th: #Holding portion
            return g1
        else: #Falling portion
            tstar = t + 0.5*(tg2 - tg1)-th
            return g1*(1-(g2f*tstar-g3)**2)*np.exp(-(g2f*tstar-g3)**2/g5)

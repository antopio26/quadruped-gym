import numpy as np
from ahrs.filters import Madgwick
from ahrs.common import Quaternion
from gymnasium import spaces

from .walking_quad import WalkingQuadrupedEnv

class POWalkingQuadrupedEnv(WalkingQuadrupedEnv):

    def __init__(self, obs_window=1, **kwargs):
        super(POWalkingQuadrupedEnv, self).__init__(**kwargs)

        # Initialize observation window
        self.obs_window = obs_window
        self.observation_buffer = []

        # Initialize Madgwick filter for orientation estimation
        self.madgwick_filter = Madgwick(Dt=self.model.opt.timestep * self.frame_skip)
        self.computed_orientation = np.array([1., 0., 0., 0.])

        # Redefine observation space to include control inputs and mask some original observations
        obs_size = 6 # Acceleration (3) + Euler angles for orientation (3)
        obs_size += 2 # Only x and y components of body_vel (optical flow)
        obs_size += self.model.nu  # Add control inputs
        obs_size += 3  # Add velocity and heading (alpha, speed, theta)
        obs_size *= self.obs_window  # Account for stacking
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_obs(self):
        """
        Obtain observation from the simulation.
        """
        accel = self._get_vec3_sensor(self._body_accel_idx)
        gyro = self._get_vec3_sensor(self._body_gyro_idx)

        # Compute the orientation using the Madgwick filter and IMU data
        self.computed_orientation = self.madgwick_filter.updateIMU(
            self.computed_orientation,
            gyr=gyro,
            acc=accel
        )

        # Convert to euler angles
        euler_angles = Quaternion(self.computed_orientation).to_angles()

        # Get the control inputs
        alpha, speed = self.control_inputs.get_velocity_aplha_speed()
        theta = self.control_inputs.get_heading_theta()

        obs = np.concatenate([
            # gyro
            accel,
            euler_angles,
            self._get_vec3_sensor(self._body_vel_idx)[0:2],  # Only x and y components (optical flow)
            self.data.ctrl,
            [alpha, speed, theta]
        ])
        return obs

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        """
        observation, info = super().reset(seed=seed, options=options)

        self.observation_buffer = [observation] * self.obs_window
        stacked_obs = np.concatenate(self.observation_buffer)

        return stacked_obs, info

    def step(self, action):
        """
        Apply the given action, advance the simulation, and return the observation, reward, done, truncated, and info.
        """
        # Step the simulation
        observation, reward, terminated, truncated, info = super().step(action)

        # Update the observation buffer
        self.observation_buffer.append(observation)

        if len(self.observation_buffer) > self.obs_window:
            self.observation_buffer.pop(0)
        elif len(self.observation_buffer) < self.obs_window:
            # Fill the rest of the previous observations with copies the current observation
            self.observation_buffer = [self.observation_buffer[0]] * (self.obs_window - len(self.observation_buffer)) + self.observation_buffer

        stacked_obs = np.concatenate(self.observation_buffer)
        return stacked_obs, reward, terminated, truncated, info
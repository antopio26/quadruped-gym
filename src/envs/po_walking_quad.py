import numpy as np
from gymnasium import spaces

from .walking_quad import WalkingQuadrupedEnv

class POWalkingQuadrupedEnv(WalkingQuadrupedEnv):

    def __init__(self, obs_window=1, **kwargs):
        super(POWalkingQuadrupedEnv, self).__init__(**kwargs)

        # Initialize observation window
        self.obs_window = obs_window
        self.observation_buffer = []

        # Redefine observation space to include control inputs and mask some original observations
        obs_size = 6  # Example: only using body_accel and body_gyro (3 each)
        obs_size += 2 # Only x and y components of body_vel (optical flow)
        obs_size += self.model.nu  # Add control inputs
        obs_size += 6  # Add velocity and heading vectors (3 each)
        obs_size *= self.frame_window  # Account for stacking
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_obs(self):
        """
        Obtain observation from the simulation.
        """
        # Example: only using body_accel and body_gyro
        obs = np.concatenate([
            self._get_vec3_sensor(self._body_accel_idx),
            self._get_vec3_sensor(self._body_gyro_idx),
            self._get_vec3_sensor(self._body_vel_idx)[0:2],  # Only x and y components (optical flow)
            self.data.ctrl,
            self.control_inputs.velocity,
            self.control_inputs.heading
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

        if len(self.observation_buffer) > self.frame_window:
            self.observation_buffer.pop(0)
        elif len(self.observation_buffer) < self.frame_window:
            # Fill the rest of the previous observations with copies the current observation
            self.observation_buffer = [self.observation_buffer[0]] * (self.frame_window - len(self.observation_buffer)) + self.observation_buffer

        stacked_obs = np.concatenate(self.observation_buffer)
        return stacked_obs, reward, terminated, truncated, info
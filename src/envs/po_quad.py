import numpy as np
from ahrs.filters import Madgwick
from ahrs.common import Quaternion
from gymnasium import spaces

from .base_quad import QuadrupedEnv

class POQuadrupedEnv(QuadrupedEnv):

    def __init__(self, obs_window=1, **kwargs):
        super(POQuadrupedEnv, self).__init__(**kwargs)

        # Validate obs_window
        if obs_window < 1:
            raise ValueError("obs_window must be greater than or equal to 1")

        # Initialize observation window
        self.obs_window = obs_window

        # Initialize Madgwick filter for orientation estimation
        self.madgwick_filter = Madgwick(Dt=self.model.opt.timestep * self.frame_skip)
        self.computed_orientation = np.array([1., 0., 0., 0.])

        # Redefine observation space to include control inputs and mask some original observations
        obs_size = 9  # Gyroscope (3) + Acceleration (3) + Euler angles for orientation (3)
        obs_size += 2  # Only x and y components of body_vel (optical flow)
        obs_size += self.model.nu  # Add control inputs (12)
        obs_size += 3  # Add velocity and heading (vx, vy, theta)    
        obs_size *= self.obs_window  # Account for stacking

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        def _get_obs(self):
            """
            Obtain observation from the simulation and handle observation stacking.
            """
            accel = self._get_vec3_sensor(self._body_accel_idx)
            gyro = self._get_vec3_sensor(self._body_gyro_idx)
        
            # Wait settling time before starting orientation integration
            if self.data.time > self.settling_time / 2:
                # Compute the orientation using the Madgwick filter and IMU data
                self.computed_orientation = self.madgwick_filter.updateIMU(
                    self.computed_orientation,
                    gyr=gyro,
                    acc=accel
                )
        
            # Convert to euler angles
            euler_angles = Quaternion(self.computed_orientation).to_angles()

            # Initialize external control observation
            external_control_obs = []

            # If external control inputs are available append to the observation
            if self.control_inputs:
                external_control_obs = self.control_inputs.get_obs()
        
            # Create the current observation
            current_obs = np.concatenate([
                gyro,
                accel,
                euler_angles,
                self._get_vec3_sensor(self._body_vel_idx)[:2],  # Only x and y components (optical flow)
                self.data.ctrl,
                external_control_obs
            ])
        
            # Handle observation stacking
            if self.obs_window == 1:
                return current_obs
            else:
                # Maintain a buffer for stacking observations
                if not hasattr(self, "_obs_buffer"):
                    self._obs_buffer = np.zeros((self.obs_window, len(current_obs)))
        
                # Shift the buffer and add the new observation
                self._obs_buffer = np.roll(self._obs_buffer, shift=-1, axis=0)
                self._obs_buffer[-1] = current_obs
        
                # Flatten the stacked observations
                return self._obs_buffer.flatten()
            
    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        Also resets the observation buffer for stacking and initializes the Madgwick filter.
        """
        # Call the parent class reset to reset the simulation
        _, info = super().reset(seed=seed, options=options)
    
        # Clear the observation buffer
        if self.obs_window > 1:
            self._obs_buffer = np.zeros((self.obs_window, self.observation_space.shape[0] // self.obs_window))
        else:
            self._obs_buffer = None
    
        # Reset the Madgwick filter and orientation
        self.computed_orientation = self._get_vec3_sensor(self._)
        self.madgwick_filter = Madgwick(Dt=self.model.opt.timestep * self.frame_skip)
    
        # Return the initial observation
        initial_observation = self._get_obs()
        return initial_observation, info
import numpy as np

from src.envs.base_quad import QuadrupedEnv
from src.controls.velocity_heading_controls import VelocityHeadingControls
from src.utils.math import exp_dist, unit, OnlineFrequencyAmplitudeEstimation

class WalkingQuadrupedEnv(QuadrupedEnv):

    def __init__(self, **kwargs):
        super(WalkingQuadrupedEnv, self).__init__(**kwargs)

        # Initialize control inputs
        self.set_external_control(VelocityHeadingControls())

        # Initialize settling time
        self.ideal_position = np.array([0.0, 0.0, 0.0], dtype=np.float32) # TODO: Generalize

        # Joint centers
        hip_center = 0.0
        knee_center = 0.0
        ankle_center = - 0.5
        self.joint_centers = np.array([hip_center, knee_center, ankle_center] * 4, dtype=np.float32)

        # Initialize previous control inputs
        self.previous_ctrl = np.copy(self.data.ctrl)

        # Initialize previous rewards to derive
        self.previous_rewards_to_derive = None

        # Initialize frequency and amplitude estimator for actuation
        self.frequency_amplitude_estimator = OnlineFrequencyAmplitudeEstimation(
            n_channels = 12,
            dt = self.model.opt.timestep * self.frame_skip,
            min_freq = 1, # Hz
            ema_alpha = 0.80,
        )

        self.ctrl_f_est = np.zeros(12, dtype=np.float32)
        self.ctrl_a_est = np.zeros(12, dtype=np.float32)

        # --- Register the reward function ---
        # Use a helper function that returns the component dictionary
        self.reward_fns = {"walking_rewards": self._get_reward_components}

        # --- Define reward keys dynamically ---
        # Call the function once to get the keys (values will be incorrect initially, but keys are set)
        initial_components = self._get_reward_components()
        self.reward_keys = list(initial_components.keys())

        # Keep self.info for potential internal debugging if needed
        self.info = {}

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.global_velocity, [1, 0, 0, 1], offset=0.1)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
        # Render the ideal position point in blue
        self.render_point(self.ideal_position, [1, 0, 1, 1])

    def compute_ideal_position(self):
        """
        Compute the ideal position based on the control inputs.
        """
        # Integrate velocity to get the ideal position
        self.ideal_position += self.control_inputs.global_velocity * self.model.opt.timestep * self.frame_skip
        return self.ideal_position

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        """
        # Call the parent class reset to reset the simulation
        observation, info = super().reset(seed=seed, options=options)

        # Reset the ideal position
        self.ideal_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # TODO: Generalize

        # Initialize previous control inputs
        self.previous_ctrl = np.copy(self.data.ctrl)

        # Initialize previous rewards to derive
        self.previous_rewards_to_derive = None

        # Reset the frequency and amplitude estimator
        # self.frequency_amplitude_estimator.reset() # Consider if reset is needed

        # Re-initialize reward keys in case reset options change behavior (optional but safer)
        initial_components = self._get_reward_components()
        self.reward_keys = list(initial_components.keys())
        self.info = {} # Reset internal info cache

        return observation, info

    def step(self, action):
        """
        Apply the given action, advance the simulation, and return the observation, reward, done, truncated, and info.
        """
        # Update the ideal position
        self.compute_ideal_position()

        # Update frequency and amplitude estimator
        self.ctrl_f_est, self.ctrl_a_est = self.frequency_amplitude_estimator.update(self.data.ctrl)

        # Store current control for next step's difference calculation
        current_ctrl = np.copy(self.data.ctrl)

        # Step the simulation - This will call self._get_reward_components via self.reward_fns
        observation, reward, terminated, truncated, info = super().step(action)

        # Update previous control inputs *after* step uses the old self.previous_ctrl
        self.previous_ctrl = current_ctrl

        # Update internal info cache (optional, info dict from super().step should be correct now)
        # self.info = {k: info[k] for k in self.reward_keys if k in info}

        return observation, reward, terminated, truncated, info

    ### Reward Functions ###

    def ideal_position_cost(self):
        """
        Reward based on the distance from the current position to the ideal position.
        """
        current_position = self._get_vec3_sensor(self._body_pos_idx)
        distance = np.linalg.norm(current_position[:2] - self.ideal_position[:2])

        return distance  # Negative reward for larger distances

    def progress_direction_reward_local(self):
        """
        Reward for moving in the right direction (local velocity).
        """
        return np.dot(unit(self._get_vec3_sensor(self._body_vel_idx)[:2]), unit(self.control_inputs.velocity[:2]))

    def progress_speed_reward_local(self):
        """
        Reward for moving with the right speed (local velocity).
        """
        actual_vel = np.linalg.norm(self._get_vec3_sensor(self._body_vel_idx)[:2])
        input_vel = np.linalg.norm(self.control_inputs.velocity[:2])

        return actual_vel - np.square(input_vel - actual_vel)

    def heading_reward(self):
        """
        Reward for facing the right direction.
        """
        return np.dot(self._get_vec3_sensor(self._body_xaxis_idx)[:2], self.control_inputs.heading[:2])

    def orientation_reward(self):
        """
        Reward for staying upright.
        """
        return self._get_vec3_sensor(self._body_zaxis_idx)[2]

    def body_height_cost(self, height=0.12):
        """
        Reward based on the distance from the current height to the ideal height.
        """
        return np.abs(self._get_vec3_sensor(self._body_pos_idx)[2] - height) # 0.12 is the default height

    def joint_posture_cost(self):
        """
        Reward for keeping the joints in a certain posture.
        """
        return np.sum(np.square(self.data.ctrl - self.joint_centers)) / self.model.nu

    def control_cost(self):
        """
        Reward for avoiding large control inputs.
        """
        # Calculate the difference between current and previous control inputs
        control_diff = self.data.ctrl - self.previous_ctrl
        # Penalize large differences
        return np.sum(np.square(control_diff))

    def control_frequency_cost(self, target_frequencies = [1.0, 1.0, 0.0]):
        """
        Reward for avoiding large control frequencies.
        """
        target = np.array(target_frequencies * 4, dtype=np.float32) # TODO: Avoid doing it every time
        return np.sum(np.square(self.ctrl_f_est - target))  / self.model.nu

    def control_amplitude_cost(self, target_amplitudes = [1.5, 0.5, 0.0]):
        """
        Reward for targeting a specific control amplitude.
        """
        target = np.array(target_amplitudes * 4, dtype=np.float32) # TODO: Avoid doing it every time
        return np.sum(np.square(self.ctrl_a_est - target)) / self.model.nu

    def alive_bonus(self):
        """
        Reward for staying alive.
        """
        return 1

    def _get_reward_components(self):
        """
        Calculates and returns a dictionary of all reward components.
        This function will be called by the base environment's step method.
        """
        # Calculate rewards to derive *first*
        rewards_to_derive_values = np.array([
            -20.0 * self.ideal_position_cost()
            # Add other rewards to derive here if needed
        ])

        if self.previous_rewards_to_derive is None:
            # Initialize on the first step after reset
            derived_rewards_values = np.zeros_like(rewards_to_derive_values)
            self.previous_rewards_to_derive = rewards_to_derive_values
        else:
            dt = self.model.opt.timestep * self.frame_skip
            derived_rewards_values = (rewards_to_derive_values - self.previous_rewards_to_derive) / dt
            # Update for the next step
            self.previous_rewards_to_derive = rewards_to_derive_values

        # --- Define the components ---
        components = {
            'alive_bonus': +1.0 * self.alive_bonus(),
            'control_cost': -2.0 * self.control_cost(), # Uses self.previous_ctrl correctly now
            'diff_ideal_position_cost': derived_rewards_values[0], # Access the calculated derivative
            'orientation_reward': +5.0 * exp_dist(self.orientation_reward()),
            'heading_reward': +3.0 * exp_dist(self.heading_reward()),
        }

        # heading * progress ?
        # TODO: Map rewards to 0-1 range to make them easly multibiable !!!

        # Update internal info cache (optional)
        # self.info = components.copy()

        return components
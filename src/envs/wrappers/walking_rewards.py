# src/envs/wrappers/walking_rewards.py
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# Assuming VelocityHeadingControls exists and has global_velocity, heading attributes
from src.controls.velocity_heading_controls import VelocityHeadingControls
from src.utils.math import exp_dist, OnlineFrequencyAmplitudeEstimation
from src.envs.wrappers.control_input import ControlInputWrapper # Needed for type checking
from src.envs.wrappers.utils import find_wrapper_by_name

class WalkingRewardWrapper(gym.Wrapper):
    """
    Adds walking-specific rewards and termination conditions to a quadruped env.

    Assumes the environment is already wrapped with ControlInputWrapper providing
    access to control logic (like VelocityHeadingControls). Calculates rewards for
    tracking ideal position, heading, orientation, posture, control effort, etc.
    """

    class_name = "WalkingRewardWrapper"

    def __init__(self,
                 env: gym.Env,
                 target_joint_posture: Optional[np.ndarray] = None,
                 target_ctrl_frequencies: Optional[np.ndarray] = None,
                 target_ctrl_amplitudes: Optional[np.ndarray] = None):
        """
        Args:
            env: The environment to wrap (must provide ControlInputWrapper interface).
            target_joint_posture: Target joint angles for posture reward.
            target_ctrl_frequencies: Target frequencies for control signal reward.
            target_ctrl_amplitudes: Target amplitudes for control signal reward.
        """
        super().__init__(env)

        # --- Get control logic using get_wrapper_by_name ---
        # Search the wrapper stack for the ControlInputWrapper instance
        control_wrapper = find_wrapper_by_name(self, "ControlInputWrapper")

        # Check if the wrapper was found
        if control_wrapper is None:
            raise TypeError("WalkingRewardWrapper requires the environment stack "
                            "to include a ControlInputWrapper (which must have class_name='ControlInputWrapper').")

        # Access the property via the found wrapper instance
        # Ensure the found wrapper actually has the 'current_controls' property
        if not hasattr(control_wrapper, 'current_controls'):
             raise AttributeError("Found ControlInputWrapper but it lacks the 'current_controls' property.")
        self.control_logic = control_wrapper.current_controls
        # --- End of getting control logic ---

        # --- Task-Specific State ---
        self.ideal_position = np.zeros(3, dtype=np.float64) # Integrated target position
        # Initialize previous and current control states
        action_shape = self.env.action_space.shape[0]
        self.previous_ctrl = np.zeros(action_shape, dtype=np.float32)
        self.current_ctrl = np.zeros(action_shape, dtype=np.float32)
        self.previous_rewards_to_derive = None # For derivative-based rewards

        # --- Reward configuration ---
        initial_components = self._calculate_reward_components()
        self.reward_keys = list(initial_components.keys())

        # Target posture (default or provided)
        if target_joint_posture is None:
            hip, knee, ankle = 0.0, 0.0, -0.5
            self.joint_centers = np.array([hip, knee, ankle] * 4, dtype=np.float32)
        else:
            if target_joint_posture.shape != (self.env.action_space.shape[0],):
                 raise ValueError(f"target_joint_posture shape mismatch. Expected ({self.env.action_space.shape[0]},), got {target_joint_posture.shape}")
            self.joint_centers = target_joint_posture.astype(np.float32)

        # Frequency/Amplitude Estimation and Targets
        self.frequency_amplitude_estimator = OnlineFrequencyAmplitudeEstimation(
            n_channels=self.env.action_space.shape[0],
            dt=self.env.unwrapped.get_dt(), # Access base env method
            min_freq=1, ema_alpha=0.80
        )
        self.ctrl_f_est = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.ctrl_a_est = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        # Default or provided targets for freq/amp rewards
        self._target_frequencies = target_ctrl_frequencies if target_ctrl_frequencies is not None else np.array([1.0, 1.0, 0.0] * 4, dtype=np.float32)
        self._target_amplitudes = target_ctrl_amplitudes if target_ctrl_amplitudes is not None else np.array([1.5, 0.5, 0.0] * 4, dtype=np.float32)
        # Validate shapes
        if self._target_frequencies.shape != (self.env.action_space.shape[0],): raise ValueError("target_ctrl_frequencies shape mismatch")
        if self._target_amplitudes.shape != (self.env.action_space.shape[0],): raise ValueError("target_ctrl_amplitudes shape mismatch")

    # --- End of added property ---
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and reward-specific state."""
        observation, info = self.env.reset(seed=seed, options=options) # Resets ControlInputWrapper too

        # Reset wrapper state based on state *after* env.reset()
        self.ideal_position = self.env.unwrapped.get_body_position().astype(np.float64)
        # Get initial control state after reset
        initial_ctrl = self.env.unwrapped.get_control_inputs()
        self.previous_ctrl = initial_ctrl.copy()
        self.current_ctrl = initial_ctrl.copy()
        self.previous_rewards_to_derive = None
        self.frequency_amplitude_estimator.reset()
        self.ctrl_f_est.fill(0.0)
        self.ctrl_a_est.fill(0.0)

        # Info dict from env.reset() already contains control_inputs_obs
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps the environment, calculates walking rewards, and checks termination."""
        # --- Pre-step updates for reward calculation ---
        self.compute_ideal_position() # Update target based on current command
        ctrl_before_step = self.env.unwrapped.get_control_inputs() # For control cost and estimator
        self.ctrl_f_est, self.ctrl_a_est = self.frequency_amplitude_estimator.update(ctrl_before_step)

        # --- Step the wrapped environment(s) ---
        # This calls ControlInputWrapper.step() then BaseQuadrupedEnv.step()
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        # Note: base_reward is 0 from QuadrupedEnv

        # --- Calculate Reward Components ---
        reward_components = self._calculate_reward_components()
        total_reward = sum(reward_components.values())

        # TODO: Rewrite termination logic

        # --- Check Walking-Specific Termination ---
        # Terminate if fallen over (check Z-axis of body)
        # Use Z-axis sensor for robustness against slight ground penetration
        # if self.env.unwrapped.get_body_z_axis()[2] < self.fall_termination_height:
            # terminated = True
            # Optionally add a penalty for falling
            # total_reward -= 10.0 # Example penalty
            # reward_components['fall_penalty'] = -10.0

        # Update info dictionary with reward components
        info.update(reward_components)
        info['ideal_position'] = self.ideal_position.copy() # Add tracking target to info

        # --- Update state for next step's reward calculation ---
        self.previous_ctrl = ctrl_before_step # Use control state active during this step

        return observation, total_reward, terminated, truncated, info

    def compute_ideal_position(self):
        """Computes the ideal position by integrating the commanded global velocity."""
        # Assumes control_logic provides global_velocity attribute
        if hasattr(self.control_logic, 'global_velocity'):
            global_velocity = self.control_logic.global_velocity
            dt = self.env.unwrapped.get_dt()
            self.ideal_position += global_velocity * dt
        else:
             print("Warning: Control logic does not have 'global_velocity'. Ideal position not updated.")

    # --- Reward Component Functions (using public accessors via unwrapped env) ---

    def _ideal_position_cost(self) -> float:
        """Calculates the cost based on the distance from the ideal position."""
        current_position = self.env.unwrapped.get_body_position()
        distance = np.linalg.norm(current_position[:2] - self.ideal_position[:2])
        return float(distance)

    def _heading_reward(self) -> float:
        """Calculates the reward based on the heading direction."""
        if not hasattr(self.control_logic, 'heading'): return 0.0
        
        # TODO: Remove unnecessary normalization
        body_x_axis_xy = self.env.unwrapped.get_body_x_axis()[:2]
        heading_xy = self.control_logic.heading[:2]
        norm_body_x = np.linalg.norm(body_x_axis_xy)
        norm_heading = np.linalg.norm(heading_xy)
        
        if norm_body_x < 1e-6 or norm_heading < 1e-6: return 0.0
        
        return float(np.dot(body_x_axis_xy / norm_body_x, heading_xy / norm_heading))

    def _orientation_reward(self) -> float:
        """Calculates the reward based on the orientation."""
        return float(self.env.unwrapped.get_body_z_axis()[2])

    def _body_height_cost(self, target_height: float = 0.12) -> float:
        """Calculates the cost based on the distance from the target height."""
        current_height = self.env.unwrapped.get_body_position()[2]
        return float(np.abs(current_height - target_height))

    def _joint_posture_cost(self) -> float:
        """Calculates the cost based on the difference from the target posture."""
        current_joint_angles = self.env.unwrapped.get_joint_angles()
        cost = np.sum(np.square(current_joint_angles - self.joint_centers))
        return float(cost / self.env.action_space.shape[0]) if self.env.action_space.shape[0] > 0 else 0.0

    def _control_cost(self) -> float:
        """Calculates the cost based on the change in control inputs from the previous step."""
        # Compares controls applied in this step (self.current_ctrl) vs previous (self.previous_ctrl)
        control_diff = self.current_ctrl - self.previous_ctrl
        cost = np.sum(np.square(control_diff))
        return float(cost)

    def _control_frequency_cost(self) -> float:
        """Calculates the cost based on the difference from the target frequencies."""
        cost = np.sum(np.square(self.ctrl_f_est - self._target_frequencies))
        return float(cost / self.env.action_space.shape[0]) if self.env.action_space.shape[0] > 0 else 0.0

    def _control_amplitude_cost(self) -> float:
        """Calculates the cost based on the difference from the target amplitudes."""
        cost = np.sum(np.square(self.ctrl_a_est - self._target_amplitudes))
        return float(cost / self.env.action_space.shape[0]) if self.env.action_space.shape[0] > 0 else 0.0

    def _alive_bonus(self) -> float:
        return 1.0

    # --- Reward Aggregation ---

    def _calculate_reward_components(self) -> Dict[str, float]:
        """
        Calculates and returns a dictionary of all reward components.
        Relies on instance attributes like self.current_ctrl being set correctly beforehand.
        """
        # Calculate rewards to derive *first*
        rewards_to_derive_values = np.array([
            -20.0 * self._ideal_position_cost()
            # Add other rewards to derive here if needed
        ])

        if self.previous_rewards_to_derive is None:
            # Initialize on the first step after reset
            derived_rewards_values = np.zeros_like(rewards_to_derive_values)
            self.previous_rewards_to_derive = rewards_to_derive_values.copy() # Use copy
        else:
            # <<< TODO: Fix dt calculation - self.model and self.frame_skip might not exist here
            # dt = self.model.opt.timestep * self.frame_skip # This likely needs fixing
            dt = self.env.unwrapped.get_dt() # Safer way to get dt
            derived_rewards_values = (rewards_to_derive_values - self.previous_rewards_to_derive) / dt
            # Update for the next step
            self.previous_rewards_to_derive = rewards_to_derive_values.copy() # Use copy

        # --- Define the components ---
        components = {
            'alive_bonus': +1.0 * self._alive_bonus(),
            'control_cost': -2.0 * self._control_cost(),
            'diff_ideal_position_cost': derived_rewards_values[0], # Access the calculated derivative
            'orientation_reward': +5.0 * exp_dist(self._orientation_reward()),
            'heading_reward': +3.0 * exp_dist(self._heading_reward()),
            # --- Example: Add other costs back if needed ---
            # 'body_height_cost': -10.0 * self._body_height_cost(),
            # 'joint_posture_cost': -1.0 * self._joint_posture_cost(),
            # 'control_frequency_cost': -0.5 * self._control_frequency_cost(),
            # 'control_amplitude_cost': -0.5 * self._control_amplitude_cost(),
        }

        return components
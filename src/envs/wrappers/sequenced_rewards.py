# src/envs/wrappers/walking_rewards.py
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from src.utils.math_utils import exp_dist
from src.utils.envs import find_wrapper_by_name
from src.utils.rewards import tolerance
from src.envs.base_quad import QuadrupedEnv

def dot_product_tolerance(x):
    """Returns the tolerance of the dot product between two vectors."""
    return tolerance(
        x,
        bounds=(1.0, np.inf),
        margin=2.0,
        value_at_margin=0.0,
        sigmoid='linear' # Use cosine or linear for finite support
    )

class SequencesRewardWrapper(gym.Wrapper):
    """
    Adds walking-specific sequenced rewards conditions to a quadruped env.

    
    
    """

    class_name = "SequencedRewardWrapper"

    def __init__(self,
                 env: gym.Env):
        """
        Args:
            env: The environment to wrap (must provide ControlInputWrapper interface).
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
        # Initialize previous and current control states
        action_shape = self.env.action_space.shape[0]
        self.previous_ctrl = np.ones(action_shape, dtype=np.float32)
        self.current_ctrl = np.zeros(action_shape, dtype=np.float32)

        # --- Reward configuration ---
        initial_components = self._calculate_reward_components()
        self.reward_keys = list(initial_components.keys())
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and reward-specific state."""
        observation, info = self.env.reset(seed=seed, options=options) # Resets ControlInputWrapper too

        # Reset wrapper state based on state *after* env.reset()
        # Get initial control state after reset
        initial_ctrl = self.env.unwrapped.get_control_inputs()
        self.previous_ctrl = np.zeros_like(initial_ctrl, dtype=np.float32) # Reset previous control
        self.current_ctrl = initial_ctrl.copy()      

        # Info dict from env.reset() already contains control_inputs_obs
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps the environment, calculates walking rewards, and checks termination."""
        # --- Pre-step updates for reward calculation ---
        ctrl_before_step = self.env.unwrapped.get_control_inputs() # For control cost and estimator
        
        # --- Step the wrapped environment(s) ---
        # This calls ControlInputWrapper.step() then BaseQuadrupedEnv.step()
        observation, base_reward, terminated, truncated, info = self.env.step(action)

        # --- Calculate Reward Components ---
        reward_components = self._calculate_reward_components()
        total_reward = sum(reward_components.values()) / len(reward_components) # Average reward

        # Update info dictionary with reward components
        info.update(reward_components)
    
        # --- Update state for next step's reward calculation ---
        self.previous_ctrl = ctrl_before_step # Use control state active during this step

        return observation, total_reward, terminated, truncated, info

    # --- Reward Component Functions (using public accessors via unwrapped env) ---

    def _heading_reward(self) -> float:
        """Calculates the reward based on the heading direction."""
        if not hasattr(self.control_logic, 'heading'): return 0.0
        
        body_x_axis_xy = self.env.unwrapped.get_body_x_axis()[:2]
        heading_xy = self.control_logic.heading[:2]
        
        return float(dot_product_tolerance(np.dot(body_x_axis_xy, heading_xy)))

    def _body_height_reward(self, target_height: float = 0.12) -> float:
        """Calculates the cost based on the distance from the target height."""
        current_height = self.env.unwrapped.get_body_position()[2]
        diff = np.abs(current_height - target_height)

        mapped_diff = tolerance(
            diff,
            bounds=(-np.inf, 0.0),
            margin=0.02,
            value_at_margin=0.1,
            sigmoid='gaussian'
        )

        return float(mapped_diff)

    def _orientation_reward(self) -> float:
        """Calculates the reward based on the orientation."""
        zaxis = dot_product_tolerance(self.env.unwrapped.get_body_z_axis()[2])
        height_reward = self._body_height_reward()

        return float(0.2 * zaxis + 0.8 * zaxis * height_reward)

    def _control_cost(self) -> float:
        """Calculates the cost based on the magnitude of joint velocities."""
        joint_velocities = self.env.unwrapped.get_joint_velocities()
        cost = np.sum(np.square(joint_velocities))
        # Normalize by number of joints
        num_joints = len(joint_velocities)
        if num_joints > 0:
            cost /= num_joints

        # This cost needs to be mapped to a reward (0 to 1),
        # where lower cost is better (higher reward).
        # We apply the tolerance mapping in _calculate_reward_components.
        return float(cost) # Return the raw cost here

    def _velocity_reward(self) -> float:
        """Calculates the reward based on the velocity."""
        # Get the current velocity of the body
        vel = self.env.unwrapped.get_body_linear_velocity()[:2] # Get only x and y components
        vel_norm = np.linalg.norm(vel)

        # Get target velocity
        target_vel = self.control_logic.velocity[:2] # Get only x and y components
        target_vel_norm = np.linalg.norm(target_vel)

        # Calculate direction component
        direction_reward = np.dot(vel, target_vel) / (vel_norm * target_vel_norm) if vel_norm > 0 and target_vel_norm > 0 else 1.0
        mapped_direction_reward = dot_product_tolerance(direction_reward)

        # Calculate speed component
        speed_reward = np.abs(vel_norm - target_vel_norm) / (target_vel_norm) if target_vel_norm > 0 else vel_norm
        mapped_speed_reward = tolerance(
            speed_reward,
            bounds=(-np.inf, 0.0),
            margin=0.5,
            value_at_margin=0.05,
            sigmoid='gaussian'
        )

        # Combine direction and speed rewards
        combined_reward = ( 0.3 * mapped_direction_reward +
                            0.7 * mapped_direction_reward * mapped_speed_reward )
        
        return float(combined_reward)
    
    def _speed_reward(self) -> float:
        """Calculates the reward based on the speed."""
        # Get the current velocity of the body
        vel = self.env.unwrapped.get_body_linear_velocity()[:2] # Get only x and y components
        vel_norm = np.linalg.norm(vel)

        # Get target velocity
        target_vel = self.control_logic.velocity[:2] # Get only x and y components
        target_vel_norm = np.linalg.norm(target_vel)

        # Calculate speed component
        speed_reward = tolerance(
            vel_norm,
            bounds=(target_vel_norm, target_vel_norm),
            margin=target_vel_norm,
            value_at_margin=0,
            sigmoid='linear'
        )

        return float(speed_reward)

    # --- Reward Aggregation ---

    def _calculate_reward_components(self) -> Dict[str, float]:
        """
        Calculates and returns a dictionary of all reward components.
        Relies on instance attributes like self.current_ctrl being set correctly beforehand.
        """

        # --- Map the reward components between 0 and 1 using dm_control tolerance function ---
        control_cost = tolerance(
            self._control_cost(),
            bounds=(-np.inf, 0.1),
            margin=0.5,
            value_at_margin=0.1,
            sigmoid='long_tail'
        )

        orientation_reward = self._orientation_reward()     # R_o
        heading_reward = self._heading_reward()             # R_h
        velocity_reward = self._velocity_reward()           # R_v

        # --- Define the components ---
        components = {
            "orientation_step": orientation_reward,
            "heading_step": orientation_reward * heading_reward, # * control_cost,
            "velocity_step": orientation_reward * heading_reward * velocity_reward # * control_cost,
        }

        return components
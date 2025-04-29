import gymnasium as gym
import numpy as np

class VelocityActionWrapper(gym.ActionWrapper):
    """
    Wraps an environment to accept joint velocity commands instead of position commands.

    Integrates velocity commands over time to produce position targets, respecting
    joint limits and a maximum speed. Uses the last commanded position action
    as the base for integration.
    """
    def __init__(self, env: gym.Env, max_speed: float):
        """
        Initializes the VelocityActionWrapper.

        Args:
            env: The environment to wrap.
            max_speed: The maximum angular velocity (radians/second or units/second
                       depending on joint type) corresponding to an action of +/- 1.
        """
        super().__init__(env)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("This wrapper only works with Box action spaces.")

        self.max_speed = max_speed
        self._dt = self.env.unwrapped.get_dt()

        # The new action space represents velocity commands [-1, 1] scaled later
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )

        # Store the underlying environment's position limits
        self._position_low = self.env.action_space.low
        self._position_high = self.env.action_space.high

        # Initialize the last commanded position action
        # Using the middle of the range as a starting point, or zeros if bounds are infinite
        if np.all(np.isfinite(self._position_low)) and np.all(np.isfinite(self._position_high)):
             self._initial_position_action = (self._position_low + self._position_high) / 2.0
        else:
             self._initial_position_action = np.zeros_like(self.env.action_space.sample())
        self._last_position_action = np.copy(self._initial_position_action)


    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Converts a velocity action [-1, 1] to a position action.

        Args:
            action: The desired joint velocities, scaled to [-1, 1].

        Returns:
            The calculated target joint positions for the underlying environment.
        """

        # If action is None, return None to disable control
        if action is None:
            return None

        # Ensure action is within the expected [-1, 1] range
        action = np.clip(action, -1.0, 1.0)

        # Scale action to actual velocity units
        velocity_command = action * self.max_speed

        # Integrate velocity from the *last commanded position*
        delta_positions = velocity_command * self._dt

        # Calculate the raw target position based on the last command
        target_positions = self._last_position_action + delta_positions

        # Clip the target positions to the joint limits of the original environment
        clipped_target_positions = np.clip(target_positions, self._position_low, self._position_high)

        # Store the new position command for the next step's integration
        self._last_position_action = clipped_target_positions

        return clipped_target_positions

    def reset(self, **kwargs):
        """Resets the environment and the internal state of the wrapper."""
        observation, info = super().reset(**kwargs)
        # Reset the last commanded position to its initial value
        self._last_position_action = np.copy(self._initial_position_action)
        return observation, info
# src/envs/wrappers/control_input.py
import gymnasium as gym
import numpy as np
from typing import Type, Optional, Dict, Any, Tuple

# Assuming BaseControls exists and has sample(), get_obs(), obs_size, render_geoms() etc.
from src.controls.base_controls import BaseControls

class ControlInputWrapper(gym.Wrapper):
    """
    Manages external control inputs for a wrapped Gymnasium environment.

    Initializes and holds a control logic object (subclass of BaseControls),
    samples initial controls during reset, adds the control observation
    to the info dictionary during step and reset, and provides methods
    to set control commands externally. It can also render control visualizations.
    """

    class_name = "ControlInputWrapper"

    def __init__(self,
                 env: gym.Env,
                 control_logic_class: Type[BaseControls],
                 **control_kwargs):
        """
        Args:
            env: The environment to wrap.
            control_logic_class: The class implementing the control logic (e.g., VelocityHeadingControls).
            **control_kwargs: Keyword arguments passed to the control_logic_class constructor.
        """
        super().__init__(env)

        if not issubclass(control_logic_class, BaseControls):
             raise TypeError("control_logic_class must be a subclass of BaseControls")

        # Instantiate the control logic
        try:
            self.control_logic = control_logic_class(**control_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize control logic {control_logic_class.__name__}: {e}") from e

        # Store control observation size for potential use by other wrappers
        if not hasattr(self.control_logic, 'obs_size'):
             raise AttributeError(f"Control logic class {control_logic_class.__name__} must have an 'obs_size' attribute.")
        self._control_obs_size: int = self.control_logic.obs_size

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and samples initial control inputs."""
        observation, info = self.env.reset(seed=seed, options=options)

        # Sample initial controls using combined options
        combined_options = getattr(self.env.unwrapped, 'reset_options', {}).copy()
        if options is not None:
            combined_options.update(options)
        control_options = combined_options.get("control_inputs", None)

        # Update control logic state (e.g., sample new commands)
        # Pass current orientation if needed by the control logic for sampling
        current_quat = self.env.unwrapped.get_body_orientation_quat() # Access base env method
        self.control_logic.sample(options=control_options, orientation_quat=current_quat)

        # Add control observation to info
        info['control_inputs_obs'] = self.control_logic.get_obs()
        info['control_obs_size'] = self._control_obs_size

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps the environment and adds current control observation to info."""
        # --- Optional: Update control logic dynamically before stepping ---
        # E.g., if controls decay or change based on time/state
        # dt = self.env.unwrapped.get_dt()
        # self.control_logic.update(dt) # Assuming an update method exists

        # Step the wrapped environment(s)
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Add current control observation to info
        info['control_inputs_obs'] = self.control_logic.get_obs()
        info['control_obs_size'] = self._control_obs_size

        return observation, reward, terminated, truncated, info
    
    # --- Override render ---
    def render(self) -> Optional[np.ndarray]:
        """Renders the environment and adds control visualizations."""
        # Render the underlying environment(s) first.
        render_output = self.env.render()

        # Add this wrapper's custom visualizations after the base render.
        # We need access to the renderer and helper methods from the base env.
        unwrapped_env = self.env.unwrapped
        # Check if rendering is active (renderer exists) in the base environment
        if hasattr(unwrapped_env, 'renderer') and unwrapped_env.renderer is not None:
            # Check if the control logic has a rendering method
            if hasattr(self.control_logic, 'render_geoms'):
                # Get necessary info from the base environment
                origin = unwrapped_env.get_body_position()
                # Get base environment's render helper functions
                render_vector_func = getattr(unwrapped_env, 'render_vector', None)
                render_point_func = getattr(unwrapped_env, 'render_point', None)

                # Ensure the helper functions exist before calling render_geoms
                if render_vector_func and render_point_func:
                    # print("[DEBUG] ControlInputWrapper rendering control geoms") # Optional debug print
                    self.control_logic.render_geoms(
                        origin=origin,
                        render_vector_func=render_vector_func,
                        render_point_func=render_point_func
                        # Pass the renderer scene if needed by render_geoms
                        # scene=unwrapped_env.renderer.scene
                    )

        # Return the output from the underlying render call (e.g., pixel array or None)
        return render_output

    # --- Expose control logic object ---
    @property
    def current_controls(self) -> BaseControls:
         """Provides access to the underlying control logic object."""
         return self.control_logic
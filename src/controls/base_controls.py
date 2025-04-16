# src/controls/base_controls.py
import numpy as np
from typing import Callable, List, Optional, Dict, Any

class BaseControls:
    def __init__(self):
        """
        Base class for control logic modules.
        """
        self.obs_size: int = 0 # Subclasses should define their observation size

    def get_obs(self) -> np.ndarray:
        """
        Returns the observation vector representing the current control state.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def sample(self, options: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Samples new random control parameters based on provided options.
        Accepts additional kwargs which might be needed by specific implementations
        (e.g., current orientation).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def render_geoms(self,
                     origin: np.ndarray,
                     render_vector_func: Callable[[np.ndarray, np.ndarray, List[float], float, float, float], None],
                     render_point_func: Callable[[np.ndarray, List[float], float], None]) -> None:
        """
        Renders visualizations related to the control state in the MuJoCo scene.

        Args:
            origin (np.ndarray): The reference point (e.g., robot base position) from which to draw vectors.
            render_vector_func (Callable): A function provided by the environment to render an arrow vector.
                                           Expected signature: (origin, vector, color, scale, radius, offset) -> None
            render_point_func (Callable): A function provided by the environment to render a point (sphere).
                                          Expected signature: (position, color, radius) -> None
        """
        # Base implementation does nothing. Subclasses can override this.
        pass # Or raise NotImplementedError if visualization is mandatory for all controls

    # Optional: Add an update method if controls need to change over time
    # def update(self, dt: float, **kwargs) -> None:
    #     """
    #     Updates the internal state of the control logic over a timestep dt.
    #     Accepts additional kwargs which might be needed (e.g., current state).
    #     """
    #     pass

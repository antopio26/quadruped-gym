# src/controls/velocity_heading_controls.py
import numpy as np
from typing import Callable, List, Optional, Dict, Any

from .base_controls import BaseControls

class VelocityHeadingControls(BaseControls):
    """
    Class to manage high-level control inputs for a quadruped robot.
    Maintains velocity, heading, and computes a global_velocity as a rotated 3D vector (z=0).
    Provides visualization for desired velocity and heading.
    """

    def __init__(self):
        super().__init__() # Call base class constructor
        self.velocity = np.zeros(3)         # Local velocity [vx, vy, vz=0] (relative to heading)
        self.heading = np.array([1.0, 0.0, 0.0]) # Unit heading vector [cos(theta), sin(theta), 0] - Initial forward
        self.global_velocity = np.zeros(3)  # Desired global velocity (3D, with z=0)
        self.obs_size = 3  # Size of the observation space (vx, vy, theta) - Set in base class now handled via super()
        self._update_global_velocity() # Initialize global velocity based on initial heading/velocity

    def _update_global_velocity(self):
        """
        Updates the global_velocity using the heading as a unit vector.
        Computes the rotated velocity based on current local velocity and heading.
        global_velocity = R(heading) * local_velocity
        """
        vx, vy = self.velocity[0], self.velocity[1]
        cos_h, sin_h = self.heading[0], self.heading[1]
        # Rotate local velocity by heading angle to get global velocity
        self.global_velocity[0] = cos_h * vx - sin_h * vy
        self.global_velocity[1] = sin_h * vx + cos_h * vy
        self.global_velocity[2] = 0.0 # Assume planar movement control

    def set_velocity_xy(self, x: float, y: float):
        """
        Sets the x and y components of the local velocity (relative to heading)
        and updates global_velocity.
        """
        self.velocity[0] = x
        self.velocity[1] = y
        self._update_global_velocity()

    def set_velocity_speed_alpha(self, speed: float, alpha: float):
        """
        Sets the local velocity using a polar representation (speed and angle relative to heading)
        and updates global_velocity.
        """
        self.velocity[0] = speed * np.cos(alpha)
        self.velocity[1] = speed * np.sin(alpha)
        self._update_global_velocity()

    def set_orientation(self, theta: float):
        """
        Sets the heading (orientation) from a global angle theta and updates global_velocity.
        """
        self.heading[0] = np.cos(theta)
        self.heading[1] = np.sin(theta)
        self.heading[2] = 0.0 # Ensure it's planar
        self._update_global_velocity() # Global velocity depends on heading

    def get_global_velocity_alpha_speed(self) -> tuple[float, float]:
        """
        Returns the desired global velocity in polar coordinates (speed and angle).
        """
        speed = np.linalg.norm(self.global_velocity[0:2])
        alpha = np.arctan2(self.global_velocity[1], self.global_velocity[0])
        return speed, alpha

    def get_velocity_alpha_speed(self) -> tuple[float, float]:
        """
        Returns the local velocity (relative to heading) in polar coordinates (speed and angle).
        """
        speed = np.linalg.norm(self.velocity[0:2])
        alpha = np.arctan2(self.velocity[1], self.velocity[0])
        return speed, alpha

    def get_heading_theta(self) -> float:
        """
        Returns the heading angle theta.
        """
        return np.arctan2(self.heading[1], self.heading[0])

    def sample(self, options: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Sample random control inputs using the provided options dictionary.

        Args:
            options: Dictionary with sampling options. Possible keys:
                     - 'min_speed': Minimum speed for velocity sampling.
                     - 'max_speed': Maximum speed for velocity sampling.
                     - 'fixed_heading_angle': Fixed heading angle if provided.
                     - 'fixed_velocity_angle': Fixed local velocity angle (relative to heading) if provided.
                     - 'fixed_speed': Fixed speed if provided.
            **kwargs: Catches potential extra arguments (like orientation_quat from wrapper)
                      which are not used in this specific sampling logic.
        """
        if options is None:
            options = {}

        min_speed = options.get('min_speed', 0.0)
        max_speed = options.get('max_speed', 1.0)
        fixed_heading_angle = options.get('fixed_heading_angle', None)
        fixed_velocity_angle = options.get('fixed_velocity_angle', None) # Angle relative to heading
        fixed_speed = options.get('fixed_speed', None)

        # Sample heading angle if not fixed
        if fixed_heading_angle is not None:
            theta = fixed_heading_angle
        else:
            theta = np.random.uniform(-np.pi, np.pi)

        # Set orientation (updates heading and global velocity)
        self.set_orientation(theta)

        # Sample local velocity angle (relative to heading) if not fixed
        if fixed_velocity_angle is not None:
            alpha = fixed_velocity_angle
        else:
            alpha = np.random.uniform(-np.pi, np.pi)

        # Sample speed if not fixed
        if fixed_speed is not None:
            speed = fixed_speed
        else:
            speed = np.random.uniform(min_speed, max_speed)

        # Set local velocity (updates velocity and global velocity)
        self.set_velocity_speed_alpha(speed, alpha)

    def get_obs(self) -> np.ndarray:
        """Returns the control observation: [local_vx, local_vy, heading_theta]."""
        return np.concatenate([
            self.velocity[:2],          # Local velocity components
            [self.get_heading_theta()]  # Global heading angle
        ])

    def render_geoms(self,
                     origin: np.ndarray,
                     render_vector_func: Callable[[np.ndarray, np.ndarray, List[float], float, float, float], None],
                     render_point_func: Callable[[np.ndarray, List[float], float], None]) -> None:
        """
        Renders the desired global velocity and heading vectors.

        Args:
            origin (np.ndarray): The robot's base position.
            render_vector_func (Callable): Function to draw an arrow.
            render_point_func (Callable): Function to draw a point (unused here).
        """
        # --- Visualize Desired Global Velocity ---
        # Make velocity vector thicker and slightly offset upwards for visibility
        vel_color = [0.0, 0.8, 0.0, 0.8] # Green, slightly transparent
        vel_scale = 0.5 # Scale velocity magnitude directly
        vel_radius = 0.005
        vel_offset = 0.05 # Draw slightly above the origin point
        # Only draw if velocity magnitude is significant
        if np.linalg.norm(self.global_velocity) > 1e-3:
            render_vector_func(origin, self.global_velocity, vel_color, vel_scale, vel_radius, vel_offset)

        # --- Visualize Heading Direction ---
        # Draw a shorter, thinner vector for heading
        head_color = [0.8, 0.0, 0.0, 0.8] # Blue, slightly transparent
        head_scale = 0.2 # Fixed length for heading indicator
        head_radius = 0.005
        head_offset = 0.05 # Draw at the same offset as velocity
        render_vector_func(origin, self.heading, head_color, head_scale, head_radius, head_offset)

        # --- Optional: Visualize Local Velocity (relative to heading) ---
        # To visualize this, we need the robot's current orientation matrix
        # This is more complex as it requires getting the rotation matrix from the env
        # and rotating the local velocity vector into the global frame *before* rendering.
        # Example (requires access to robot's rotation matrix `rot_matrix`):
        # global_local_vel = rot_matrix @ self.velocity
        # local_vel_color = [0.8, 0.0, 0.0, 0.8] # Red
        # render_vector_func(origin, global_local_vel, local_vel_color, vel_scale, vel_radius, vel_offset + 0.01) # Slightly higher offset

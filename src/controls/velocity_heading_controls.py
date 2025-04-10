import numpy as np

from .base_controls import BaseControls

class VelocityHeadingControls(BaseControls):
    """
    Class to manage high-level control inputs for a quadruped robot.
    Maintains velocity, heading, and computes a global_velocity as a rotated 3D vector (z=0).
    """

    def __init__(self):
        self.velocity = np.zeros(3)         # Local velocity [vx, vy, vz]
        self.heading = np.zeros(3)          # Unit heading vector [cos(theta), sin(theta), _]
        self.global_velocity = np.zeros(3)  # Rotated velocity (3D, with z=0)

    def update_global_velocity(self):
        """
        Updates the global_velocity using the heading as a unit vector.
        Computes the rotated velocity as:
          global_velocity[0] = heading[0]*velocity[0] - heading[1]*velocity[1]
          global_velocity[1] = heading[1]*velocity[0] + heading[0]*velocity[1]
          global_velocity[2] = 0
        """
        v0, v1 = self.velocity[0], self.velocity[1]
        h0, h1 = self.heading[0], self.heading[1]
        self.global_velocity[0] = h0 * v0 - h1 * v1
        self.global_velocity[1] = h1 * v0 + h0 * v1
        self.global_velocity[2] = 0.0

    def set_velocity_xy(self, x, y):
        """
        Sets the x and y components of the velocity and updates global_velocity.
        """
        self.velocity[0] = x
        self.velocity[1] = y
        self.update_global_velocity()

    def set_velocity_speed_alpha(self, speed, alpha):
        """
        Sets the velocity using a polar representation (speed and angle) and updates global_velocity.
        """
        self.velocity[0] = speed * np.cos(alpha)
        self.velocity[1] = speed * np.sin(alpha)
        self.update_global_velocity()

    def set_orientation(self, theta):
        """
        Sets the heading (orientation) from an angle theta and updates global_velocity.
        """
        self.heading[0] = np.cos(theta)
        self.heading[1] = np.sin(theta)
        self.update_global_velocity()

    def get_global_velocity_alpha_speed(self):
        """
        Returns the velocity in polar coordinates (speed and angle).
        """
        speed = np.linalg.norm(self.global_velocity[0:2])
        alpha = np.arctan2(self.global_velocity[1], self.global_velocity[0])
        return speed, alpha

    def get_velocity_aplha_speed(self):
        """
        Returns the velocity in polar coordinates (speed and angle).
        """
        speed = np.linalg.norm(self.velocity[0:2])
        alpha = np.arctan2(self.velocity[1], self.velocity[0])
        return speed, alpha

    def get_heading_theta(self):
        """
        Returns the heading angle theta.
        """
        return np.arctan2(self.heading[1], self.heading[0])

    def sample(self, options=None):
        """
        Sample random control inputs using the provided options dictionary.

        :param options: Dictionary with sampling options. Possible keys:
                        - 'min_speed': Minimum speed for velocity sampling.
                        - 'max_speed': Maximum speed for velocity sampling.
                        - 'fixed_heading_angle': Fixed heading angle if provided.
                        - 'fixed_velocity_angle': Fixed velocity angle if provided.
                        - 'fixed_speed': Fixed speed if provided.
        """
        if options is None:
            options = {}

        min_speed = options.get('min_speed', 0.0)
        max_speed = options.get('max_speed', 1.0)
        fixed_heading_angle = options.get('fixed_heading_angle', None)
        fixed_velocity_angle = options.get('fixed_velocity_angle', None)
        fixed_speed = options.get('fixed_speed', None)

        # Sample heading angle if not fixed
        if fixed_heading_angle is not None:
            theta = fixed_heading_angle
        else:
            theta = np.random.uniform(-np.pi, np.pi)

        # Set orientation
        self.set_orientation(theta)

        # Sample velocity angle if not fixed
        if fixed_velocity_angle is not None:
            alpha = fixed_velocity_angle
        else:
            alpha = np.random.uniform(-np.pi, np.pi)

        # Sample speed if not fixed
        if fixed_speed is not None:
            speed = fixed_speed
        else:
            speed = np.random.uniform(min_speed, max_speed)

        # Set velocity
        self.set_velocity_speed_alpha(speed, alpha)
    
    def get_obs(self):
        return np.concatenate([
            self.velocity,
            self.heading
        ])
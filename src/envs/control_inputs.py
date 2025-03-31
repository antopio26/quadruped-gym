import numpy as np

class VelocityHeadingControls:
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

    def sample(self, min_speed=0.5, max_speed=1.0):
        """
        Randomly samples velocity and orientation, updates the control inputs,
        and returns the updated velocity, heading, and global_velocity.
        """
        speed = np.random.uniform(min_speed, max_speed)
        alpha = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        self.set_velocity_speed_alpha(speed, alpha)
        self.set_orientation(theta)
        return self.velocity, self.heading, self.global_velocity

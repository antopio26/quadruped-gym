import numpy as np

class VelocityHeadingControls:
    """
    Class to manage high-level control inputs for a quadruped robot.
    """

    ####### Also height could be controlled #######

    def __init__(self):
        self.velocity = np.zeros(3)
        self.heading = np.zeros(3)

    def set_velocity_xy(self, x, y):
        self.velocity[0] = x
        self.velocity[1] = y

    def set_velocity_speed_alpha(self, speed, alpha):
        self.velocity[0] = speed * np.cos(alpha)
        self.velocity[1] = speed * np.sin(alpha)

    def set_orientation(self, theta):
        self.heading[0] = np.cos(theta)
        self.heading[1] = np.sin(theta)

    def sample(self, max_speed=1.0):
        # Randomly sample velocity and orientation.
        self.set_velocity_speed_alpha(
            speed=np.random.uniform(0, max_speed),
            alpha=np.random.uniform(-np.pi, np.pi)
        )

        # Randomly sample orientation.
        self.set_orientation(
            theta=np.random.uniform(-np.pi, np.pi)
        )

        return self.velocity, self.heading
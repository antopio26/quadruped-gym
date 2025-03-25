import numpy as np
from mujoco import mj_name2id, mjtObj

from .quadruped import QuadrupedEnv
from .control_inputs import VelocityHeadingControls

class WalkingQuadrupedEnv(QuadrupedEnv):

    def __init__(self, **kwargs):
        super(WalkingQuadrupedEnv, self).__init__(**kwargs)

        # Useful constants for custom reward functions.
        self._body_accel_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_accel")]
        self._body_gyro_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self._body_pos_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_pos")]

        self._body_linvel_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_linvel")]
        self._body_xaxis_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_xaxis")]
        self._body_xaxis_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_xaxis")]

        self._get_vec3_sensor = lambda idx: self.data.sensordata[idx: idx+3]

        self.control_inputs = VelocityHeadingControls()

        # Initialize previous control inputs
        self.previous_ctrl = np.zeros_like(self.data.ctrl)


    def progress_direction_reward(self):
        # Reward for moving in the right direction.
        return np.dot(self._get_vec3_sensor(self._body_linvel_idx), self.control_inputs.velocity)

    def progress_speed_reward(self):
        # Reward for moving with the right speed.
        d = np.abs(np.abs(self._get_vec3_sensor(self._body_linvel_idx)) - np.abs(self.control_inputs.velocity))

        return -0.1 * np.sum(np.square(d))

    def orientation_reward(self):
        # Reward for facing the right direction.
        return np.dot(self._get_vec3_sensor(self._body_xaxis_idx), self.control_inputs.heading)

    # Other rewards based on frame position, orientation, etc.
    # (like not flipping or keeping the body upright) can be added.

    # DUMMY REWARD FUNCTION
    def control_cost(self):
        # Calculate the difference between current and previous control inputs
        control_diff = self.data.ctrl - self.previous_ctrl
        # Update previous control inputs
        self.previous_ctrl = np.copy(self.data.ctrl)
        # Penalize large differences
        return -0.1 * np.sum(np.square(control_diff))

    def alive_bonus(self):
        # Constant bonus for staying "alive".
        return 0.1

    def

    def _default_reward(self):


        return (self.alive_bonus()
                + self.control_cost()
                )

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.velocity, [1, 0, 0, 1], offset=0.05)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
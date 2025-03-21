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

    def control_cost(self):
        # Penalize high control inputs.
        return -0.1 * np.sum(np.square(self.data.ctrl))

    def alive_bonus(self):
        # Constant bonus for staying "alive".
        return 0.5 * self.frame_skip

    def progress_reward(self):
        # Reward for moving in the right direction. (Magnitude-weighted cosine similarity)
        v1 = self._get_vec3_sensor(self._body_linvel_idx)
        v2 = self.control_inputs.velocity

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 < 1e-8 or norm_v2 < 1e-8:
            return 0  # Avoid division by zero and numerical instability

        return (np.dot(v1, v2) / (norm_v1 * norm_v2)) * (min(norm_v1, norm_v2) / max(norm_v1, norm_v2))

    def orientation_reward(self):
        # Reward for facing the right direction.
        return np.dot(self._get_vec3_sensor(self._body_xaxis_idx), self.control_inputs.heading)

    # Other rewards based on frame position, orientation, etc.
    # (like not flipping or keeping the body upright) can be added.

    def _default_reward(self):
        return self.alive_bonus() + self.control_cost() + 2 * self.progress_reward() + 2 * self.orientation_reward()

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.velocity, [1, 0, 0, 1], offset=0.05)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
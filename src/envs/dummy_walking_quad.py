import numpy as np

from .diff_walking_quad import WalkingQuadrupedEnv

class DummyWalkingQuadrupedEnv(WalkingQuadrupedEnv):

    def __init__(self, **kwargs):
        super(DummyWalkingQuadrupedEnv, self).__init__(**kwargs)

    ## DUMMY REWARDS ##
    def forward_reward(self):
        # Reward for moving in the right direction.
        return self._get_vec3_sensor(self._body_linvel_idx)[0] * self._get_vec3_sensor(self._body_pos_idx)[0]

    def no_drift_reward(self):
        # Penalize movement on y axis
        return np.abs(self._get_vec3_sensor(self._body_linvel_idx)[1] * self._get_vec3_sensor(self._body_pos_idx)[1])

    def _default_reward(self):
        return (+ 0.1 * self.alive_bonus()
                - 0.5 * self.control_cost()
                + 5.0 * self.forward_reward()
                - 3.0 * self.no_drift_reward()
                )
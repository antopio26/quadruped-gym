import numpy as np
from mujoco import mj_name2id, mjtObj
from gymnasium import spaces

from .quadruped import QuadrupedEnv
from .control_inputs import VelocityHeadingControls

# Utility function for exponential rewards
def exp_dist(x):
    return np.exp(x) - 1

class WalkingQuadrupedEnv(QuadrupedEnv):

    def __init__(self, frame_window=1, random_controls=False, min_speed=0.2, max_speed=1.0, **kwargs):
        super(WalkingQuadrupedEnv, self).__init__(**kwargs)

        self.frame_window = frame_window
        self.observation_buffer = []

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.random_controls = random_controls

        # Useful constants for custom reward functions.
        self._body_accel_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_accel")]
        self._body_gyro_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self._body_pos_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_pos")]
        self._body_linvel_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_linvel")]
        self._body_xaxis_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_xaxis")]
        self._body_zaxis_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_zaxis")]
        self._body_vel_idx = self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "body_vel")]

        self._get_vec3_sensor = lambda idx: self.data.sensordata[idx: idx + 3]

        # Initialize control inputs
        self.control_inputs = VelocityHeadingControls()
        self.ideal_position = np.array([0.0,0.0,0.0], dtype=np.float64) # TODO: Generalize

        # Initialize previous control inputs
        self.previous_ctrl = np.zeros_like(self.data.ctrl)

        # Redefine observation space to include control inputs and mask some original observations
        obs_size = 6  # Example: only using body_accel and body_gyro (3 each)
        obs_size += 2 # Only x and y components of body_vel (optical flow)
        obs_size += self.model.nu  # Add control inputs
        obs_size += 6  # Add velocity and heading vectors (3 each)
        obs_size *= self.frame_window  # Account for stacking
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_obs(self):
        """Obtain observation from the simulation."""
        # Example: only using body_accel and body_gyro
        obs = np.concatenate([
            self._get_vec3_sensor(self._body_accel_idx),
            self._get_vec3_sensor(self._body_gyro_idx),
            self._get_vec3_sensor(self._body_vel_idx)[0:2],  # Only x and y components (optical flow)
            self.data.ctrl,
            self.control_inputs.velocity,
            self.control_inputs.heading
        ])
        return obs

    def reset(self, seed=None, options=None):
        """Reset the simulation to an initial state and return the initial observation."""
        observation, info = super().reset(seed=seed, options=options)
        self.observation_buffer = [observation] * self.frame_window
        stacked_obs = np.concatenate(self.observation_buffer)

        # Reset the ideal position
        self.ideal_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # TODO: Generalize

        if self.random_controls:
            self.control_inputs.sample(min_speed=self.min_speed, max_speed=self.max_speed)

        return stacked_obs, info

    def step(self, action):
        """Apply the given action, advance the simulation, and return the observation, reward, done, truncated, and info."""

        # Update the ideal position
        self.compute_ideal_position()

        # Step the simulation
        observation, reward, terminated, truncated, info = super().step(action)

        # Update the observation buffer
        self.observation_buffer.append(observation)

        if len(self.observation_buffer) > self.frame_window:
            self.observation_buffer.pop(0)
        elif len(self.observation_buffer) < self.frame_window:
            # Fill the rest of the previous observations with copies the current observation
            self.observation_buffer = [self.observation_buffer[0]] * (self.frame_window - len(self.observation_buffer)) + self.observation_buffer

        stacked_obs = np.concatenate(self.observation_buffer)
        return stacked_obs, reward, terminated, truncated, info

    def flip_termination(self):
        # Terminate the episode if the body flips upside down.
        return self._get_vec3_sensor(self._body_zaxis_idx)[2] < 0

    def _default_termination(self):
        return self.flip_termination() or super()._default_termination()

    # TODO: ACCOUT FOR HEADING VECTOR AS VELOCITY IS IN THE LOCAL FRAME
    def compute_ideal_position(self):
        """
        Compute the ideal position based on the control inputs.
        """
        # Integrate velocity to get the ideal position
        self.ideal_position += self.control_inputs.global_velocity * self.model.opt.timestep * self.frame_skip
        return self.ideal_position

    def ideal_position_cost(self):
        """
        Reward based on the distance from the current position to the ideal position.
        """
        current_position = self._get_vec3_sensor(self._body_pos_idx)
        distance = np.linalg.norm(current_position[:2] - self.ideal_position[:2])
        return distance  # Negative reward for larger distances

    def progress_direction_reward_global(self):
        # Reward for moving in the right direction.
        return np.dot(self._get_vec3_sensor(self._body_linvel_idx), self.control_inputs.velocity)

    def progress_speed_cost_global(self):
        # Reward for moving with the right speed.
        d = np.abs(self._get_vec3_sensor(self._body_linvel_idx)) - np.abs(self.control_inputs.velocity)

        return np.sum(np.square(d))

    def progress_direction_reward_local(self):
        # Reward for moving in the right direction.
        return np.dot(self._get_vec3_sensor(self._body_vel_idx), self.control_inputs.velocity)

    def progress_speed_cost_local(self):
        # Reward for moving with the right speed.
        d = np.abs(self._get_vec3_sensor(self._body_vel_idx) - self.control_inputs.velocity)

        return np.sum(np.square(d))

    def heading_reward(self):
        # Reward for facing the right direction.
        return np.dot(self._get_vec3_sensor(self._body_xaxis_idx)[:2], self.control_inputs.heading[:2])

    def orientation_reward(self):
        # Reward for staying upright.
        return self._get_vec3_sensor(self._body_zaxis_idx)[2]

    # TODO: Body height reward
    def body_height_cost(self, height=0.12):
        # Reward for keeping the body at a certain height.
        return np.abs(self._get_vec3_sensor(self._body_pos_idx)[2] - height) # 0.12 is the default height

    def joint_posture_cost(self):
        # Penalize large joint angles
        foot_center = - 0.5
        centers = np.array([0, 0, foot_center, 0, 0, foot_center, 0, 0, foot_center, 0, 0, foot_center])

        return np.sum(np.square(self.data.ctrl - centers))

    # TODO: Vibration cost

    # Other rewards based on frame position, orientation, etc.
    # (like not flipping or keeping the body upright) can be added.

    def control_cost(self):
        # Calculate the difference between current and previous control inputs
        control_diff = self.data.ctrl - self.previous_ctrl
        # Update previous control inputs
        self.previous_ctrl = np.copy(self.data.ctrl)
        # Penalize large differences
        return np.sum(np.square(control_diff))

    def alive_bonus(self):
        # Constant bonus for staying "alive".
        return 1

    # NOTE: Maybe multiply some of the rewards

    def input_control_reward(self):
        return (+ 5.0 * self.alive_bonus()
                - 1.0 * self.control_cost()
                + 10.0 * self.progress_direction_reward_local()
                - 10.0 * self.progress_speed_cost_local()
                + 5.0 * self.heading_reward()
                + 5.0 * exp_dist(self.orientation_reward())
                - 1.0 * exp_dist(self.body_height_cost())
                - 0.5 * self.joint_posture_cost()
                - 1.0 * self.ideal_position_cost()
                )

    ## DUMMY REWARD FUNCTIONS ##
    def forward_reward(self):
        # Reward for moving in the right direction.
        return self._get_vec3_sensor(self._body_linvel_idx)[0] * self._get_vec3_sensor(self._body_pos_idx)[0]

    def drift_cost(self):
        # Penalize movement on y axis
        return np.abs(self._get_vec3_sensor(self._body_linvel_idx)[1])

    def only_forward_reward(self):
        return (+ 0.1 * self.alive_bonus()
                - 2.0 * self.control_cost()
                + 5.0 * self.forward_reward()
                - 1.0 * self.drift_cost()
                )

    def _default_reward(self):
        return self.input_control_reward()

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.global_velocity, [1, 0, 0, 1], offset=0.1)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
        # Render the ideal position point in blue
        self.render_point(self.ideal_position, [1, 0, 1, 1])
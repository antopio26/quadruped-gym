import numpy as np
from mujoco import mj_name2id, mjtObj

from .quadruped import QuadrupedEnv
from .control_inputs import VelocityHeadingControls
from .math_utils import exp_dist, OnlineFrequencyAmplitudeEstimation


class WalkingQuadrupedEnv(QuadrupedEnv):

    def __init__(self, settling_time = 0, random_controls=False, random_init=False, reset_options=None, **kwargs):
        super(WalkingQuadrupedEnv, self).__init__(**kwargs)

        self.random_controls = random_controls
        self.random_init = random_init

        self.reset_options = reset_options

        self._get_sensor_idx = lambda name: self.model.sensor_adr[mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, name)]
        self._get_vec3_sensor = lambda idx: self.data.sensordata[idx: idx + 3]

        # Get sensor indices
        self._body_accel_idx = self._get_sensor_idx("body_accel")
        self._body_gyro_idx = self._get_sensor_idx("body_gyro")
        self._body_pos_idx = self._get_sensor_idx("body_pos")
        self._body_linvel_idx = self._get_sensor_idx("body_linvel")
        self._body_xaxis_idx = self._get_sensor_idx("body_xaxis")
        self._body_zaxis_idx = self._get_sensor_idx("body_zaxis")
        self._body_vel_idx = self._get_sensor_idx("body_vel")

        # Initialize control inputs
        self.control_inputs = VelocityHeadingControls()
        self.ideal_position = np.array([0.0, 0.0, 0.0], dtype=np.float32) # TODO: Generalize

        # Joint centers
        hip_center = 0.0
        knee_center = 0.0
        ankle_center = - 0.5
        self.joint_centers = np.array([hip_center, knee_center, ankle_center] * 4, dtype=np.float32)

        # Settling time
        self.settling_time = settling_time

        # Initialize previous control inputs
        self.previous_ctrl = self.joint_centers.copy()

        # Initialize previous rewards to derive
        self.previous_rewards_to_derive = None     

        # Initialize frequency and amplitude estimator for actuation
        self.frequency_amplitude_estimator = OnlineFrequencyAmplitudeEstimation(
            n_channels = 12,
            dt = self.model.opt.timestep * self.frame_skip,
            min_freq = 1 # Hz
        )

        self.ctrl_f_est = np.zeros(12, dtype=np.float32)
        self.ctrl_a_est = np.zeros(12, dtype=np.float32)

        self.info = {}

    def initialize_robot_state(self):
        """
        Randomly initialize the heading of the frame and the joint angles of the robot.
        """
        # Randomly initialize the heading of the frame around the z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        # Set quaternion to rotate along the z-axis
        self.data.qpos[3:7] = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.global_velocity, [1, 0, 0, 1], offset=0.1)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
        # Render the ideal position point in blue
        self.render_point(self.ideal_position, [1, 0, 1, 1])

    def compute_ideal_position(self):
        """
        Compute the ideal position based on the control inputs.
        """
        # Integrate velocity to get the ideal position
        self.ideal_position += self.control_inputs.global_velocity * self.model.opt.timestep * self.frame_skip
        return self.ideal_position

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        """
        if options is None:
            options = self.reset_options

        observation, info = super().reset(seed=seed, options=options)

        # Reset the ideal position
        self.ideal_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # TODO: Generalize

        # Initialize previous control inputs
        self.previous_ctrl = self.joint_centers.copy()

        # Initialize previous rewards to derive
        self.previous_rewards_to_derive = None 

        # Reset the frequency and amplitude estimator
        # self.frequency_amplitude_estimator.reset()

        # Initialize the robot state
        if self.random_init:
            self.initialize_robot_state()

        if self.random_controls:
            self.control_inputs.sample(options=options)

        self.info = {}

        return observation, self.info

    def step(self, action):
        """
        Apply the given action, advance the simulation, and return the observation, reward, done, truncated, and info.
        """
        # Update the ideal position
        self.compute_ideal_position()

        # Update frequency and amplitude estimator
        self.ctrl_f_est, self.ctrl_a_est = self.frequency_amplitude_estimator.update(self.data.ctrl)

        # self.info['ctrl_f_est'] = self.ctrl_f_est
        # self.info['ctrl_a_est'] = self.ctrl_a_est

        # Mask actions for settling time
        if self.data.time < self.settling_time:
            action = np.array(self.joint_centers)

        # Step the simulation
        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, self.info

    ### Termination Logic ###

    def flip_termination(self):
        """
        Check if the robot is flipped.
        """
        return self._get_vec3_sensor(self._body_zaxis_idx)[2] < 0

    def _default_termination(self):
        """
        Check if the robot is flipped or if the simulation has reached the maximum time step.
        """
        return self.flip_termination() or super()._default_termination()

    ### Reward Functions ###

    def ideal_position_cost(self):
        """
        Reward based on the distance from the current position to the ideal position.
        """
        current_position = self._get_vec3_sensor(self._body_pos_idx)
        distance = np.linalg.norm(current_position[:2] - self.ideal_position[:2])

        return distance  # Negative reward for larger distances

    def progress_direction_reward_global(self):
        """
        Reward for moving in the right direction (global velocity).
        """
        return np.dot(self._get_vec3_sensor(self._body_linvel_idx), self.control_inputs.velocity)

    def progress_speed_reward_global(self, max_reward=1.0):
        """
        Reward for moving with the right speed (global velocity).
        """
        d = np.linalg.norm(self._get_vec3_sensor(self._body_linvel_idx)) - np.linalg.norm(self.control_inputs.velocity)

        return max_reward - np.square(d)

    def progress_speed_cost_global(self):
        """
        Reward for moving with the right speed (global velocity).
        """
        d = np.linalg.norm(self._get_vec3_sensor(self._body_linvel_idx)) - np.linalg.norm(self.control_inputs.velocity)

        return np.square(d)

    def progress_direction_reward_local(self):
        """
        Reward for moving in the right direction (local velocity).
        """
        return np.dot(self._get_vec3_sensor(self._body_vel_idx), self.control_inputs.velocity)

    def progress_speed_reward_local(self, max_reward=1.0):
        """
        Reward for moving with the right speed (local velocity).
        """
        d = np.linalg.norm(self._get_vec3_sensor(self._body_vel_idx)) - np.linalg.norm(self.control_inputs.velocity)

        return max_reward - np.square(d)

    def progress_speed_cost_local(self):
        """
        Reward for moving with the right speed (local velocity).
        """
        d = np.linalg.norm(self._get_vec3_sensor(self._body_linvel_idx)) - np.linalg.norm(self.control_inputs.velocity)

        return np.square(d)

    def heading_reward(self):
        """
        Reward for facing the right direction.
        """
        return np.dot(self._get_vec3_sensor(self._body_xaxis_idx)[:2], self.control_inputs.heading[:2])

    def orientation_reward(self):
        """
        Reward for staying upright.
        """
        return self._get_vec3_sensor(self._body_zaxis_idx)[2]

    def body_height_cost(self, height=0.12):
        """
        Reward based on the distance from the current height to the ideal height.
        """
        return np.abs(self._get_vec3_sensor(self._body_pos_idx)[2] - height) # 0.12 is the default height

    def joint_posture_cost(self):
        """
        Reward for keeping the joints in a certain posture.
        """
        return np.linalg.norm((self.data.ctrl - self.joint_centers) / self.model.nu)

    def control_cost(self): 
        """
        Reward for avoiding large control inputs.
        """
        # Calculate the difference between current and previous control inputs
        control_diff = self.data.ctrl - self.previous_ctrl
        # Update previous control inputs
        self.previous_ctrl = np.copy(self.data.ctrl)
        # Penalize large differences
        return np.linalg.norm(control_diff / self.model.nu)

    def control_frequency_cost(self, target_frequencies = [1.0, 1.0, 0.0]):
        """
        Reward for avoiding large control frequencies.
        """
        target = np.array(target_frequencies * 4, dtype=np.float32)
        return np.linalg.norm((self.ctrl_f_est - target) / self.model.nu)

    def control_amplitude_cost(self, target_amplitudes = [1.5, 0.5, 0.0]):
        """
        Reward for targeting a specific control amplitude.
        """
        target = np.array(target_amplitudes * 4, dtype=np.float32)

        return np.linalg.norm((self.ctrl_a_est - target) / self.model.nu)

    def alive_bonus(self):
        """
        Reward for staying alive.
        """
        return 1

    # TODO: Vibration cost

    # Other rewards based on frame position, orientation, etc.
    # (like not flipping or keeping the body upright) can be added.

    # NOTE: Maybe multiply some of the rewards

    '''
    20 steps - no
    def input_control_reward(self):
        return (+ 1.0 * self.alive_bonus()
                - 2.0 * self.control_cost()
                + 10.0 * self.progress_direction_reward_local()
                - 10.0 * self.progress_speed_cost_local()
                + 5.0 * self.heading_reward()
                + 5.0 * exp_dist(self.orientation_reward())
                - 1.0 * exp_dist(self.body_height_cost())
                - 0.5 * self.joint_posture_cost()
                - 5.0 * self.ideal_position_cost()
                )
    '''

    '''
    16 steps - no
    def input_control_reward(self):
        return (+ 1.0 * self.alive_bonus()
                - 2.0 * self.control_cost()
                + 10.0 * self.progress_direction_reward_local()
                - 10.0 * self.progress_speed_cost_local()
                + 5.0 * self.heading_reward()
                + 5.0 * exp_dist(self.orientation_reward())
                - 1.0 * exp_dist(self.body_height_cost())
                - 0.5 * self.joint_posture_cost()
                - 5.0 * self.ideal_position_cost()
                - 1.0 * self.control_amplitude_cost([1.0, 1.0, 0.0]])
                - 1.0 * self.control_frequency_cost()
                )
    '''

    def input_control_reward(self):

        keys = [
            # 'alive_bonus',
            # 'control_cost',
            # 'progress_direction_reward_local',
            # 'progress_speed_cost_local',
            # 'heading_reward',
            # 'orientation_reward',
            # 'body_height_cost',
            # 'joint_posture_cost',
            # 'ideal_position_cost',
            # 'control_amplitude_cost',
            # 'control_frequency_cost',
            # 'diff_progress_direction_reward_local',
            # 'diff_progress_speed_cost_local',
            # 'diff_heading_reward',
            # 'diff_orientation_reward',
            # 'diff_body_height_cost',
            # 'diff_joint_posture_cost',
            # 'diff_ideal_position_cost',
        ]

        # TODO: Research reward shaping
        # TODO: Explore the possibility of combining reward derivatives with the current reward to improve the instant information

        # To make the derived rewards softer they should be multiplied by the reward not derived
        # (maybe not exactly but something like that)

        # Should almost all derived rewards should be soft near the correct value ?!?

        value_rewards = np.array([
            # + 2.0 * self.alive_bonus(),
            # - 0.5 * self.control_cost(),
            # + 10.0 * self.progress_direction_reward_local(),    # remove ?
            # - 10.0 * self.progress_speed_cost_local(),          # remove ?
            # + 5.0 * self.heading_reward(),                      # remove ?
            # + 5.0 * exp_dist(self.orientation_reward()),        # remove ?
            # - 5.0 * exp_dist(self.body_height_cost()),          # remove ?
            # - 0.5 * self.joint_posture_cost(), 
            # - 0.5 * self.ideal_position_cost(),                 # remove ?
            # - 3.0 * self.control_amplitude_cost(),              # Don't know how to hanldle this
            # - 1.0 * self.control_frequency_cost()               # Don't know how to hanldle this
        ])
        
        rewards_to_derive = np.array([
            # + 0.5 * self.progress_direction_reward_local(),         # 1
            # - 0.5 * self.progress_speed_cost_local(),               # 1 
            # + 2.0 * self.heading_reward(),                          # 1
            # + 2.0 * exp_dist(self.orientation_reward()),            # 1
            # - 5.0 * exp_dist(self.body_height_cost()),              # 1
            # - 0.5 * self.joint_posture_cost(),                      # 1     
            # - 20.0 * self.ideal_position_cost()                     # 10
        ])

        # TODO: When do I apply the weights?
        
        if self.previous_rewards_to_derive is None:
            # Initialize the previous rewards to derive
            self.previous_rewards_to_derive = rewards_to_derive

        # Calculate the derived rewards
        derived_rewards = (rewards_to_derive - self.previous_rewards_to_derive) / (self.model.opt.timestep * self.frame_skip)

        # Set the previous rewards to derive
        self.previous_rewards_to_derive = rewards_to_derive

        # Concatenate the rewards
        values = np.concatenate((value_rewards, derived_rewards))


        '''
        values = [
            + 1.0 * self.alive_bonus(),
            - 2.0 * self.control_cost(),
            + 10.0 * self.progress_direction_reward_local(),        # Derive
            - 10.0 * self.progress_speed_cost_local(),              # Derive
            + 5.0 * self.heading_reward(),                          # Derive
            + 5.0 * exp_dist(self.orientation_reward()),            # Derive (this should be soft near the correct orientation)
            - 1.0 * exp_dist(self.body_height_cost()),              # Derive (this should be soft near the correct height)
            - 0.5 * self.joint_posture_cost(),                      # Derive
            - 0.0 * self.ideal_position_cost(),                     # Derive
            - 2.5 * self.control_amplitude_cost(),                  # Don't know how to hanldle this
            - 3.0 * self.control_frequency_cost()                   # Don't know how to hanldle this
        ]
        '''

        # Create a dictionary of rewards
        self.info = {key: value for key, value in zip(keys, values)}

        # Sum all values
        return sum(values)

    # NOTE: I think rewards should judge an action, not a state so maybe
    #       for some of these rewards we should consider the derivative

    def _default_reward(self):
        return self.input_control_reward()
import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import mujoco.viewer

from src.controls.base_controls import BaseControls

class QuadrupedEnv(gym.Env):
    """
    Custom Gymnasium environment for a MuJoCo-based quadruped.

    This environment loads the model from an XML file and simulates the dynamics.
    It applies user-specified control inputs, steps the simulation with an optional
    frame-skip, and supports both "human" and "rgb_array" render modes with decoupled
    simulation time and rendering FPS. Additionally, video can be recorded independently
    at the specified render_fps.

    Attributes:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        max_time (float): Maximum episode duration (seconds).
        frame_skip (int): Number of simulation steps per environment step.
        render_mode (str or None): One of "human", "rgb_array", or None.
        width (int): Width of rendered images.
        height (int): Height of rendered images.
        render_fps (int): Target frames per second for rendering.
        scene_option (mujoco.MjvOption): Rendering options.
        renderer (mujoco.Renderer or None): Renderer instance (created on demand).
        reward_fns (dict): Dictionary mapping reward names to reward functions.
        termination_fns (dict): Dictionary mapping termination names to functions.
        save_video (bool): Whether to save a video of the simulation.
        video_path (str): Path to save the video file.
        video_writer (cv2.VideoWriter or None): Video writer instance.
        _sim_start_time (float or None): Wall-clock time when simulation starts.
        _frame_count (int): Number of frames rendered so far.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 model_path: str = "./models/quadruped/scene.xml",
                 max_time: float = 10.0,
                 frame_skip: int = 16,
                 render_mode: str = None,
                 width: int = 720,
                 height: int = 480,
                 render_fps: int = 30,
                 reward_fns: dict = None,
                 termination_fns: dict = None,
                 save_video: bool = False,
                 video_path: str = "videos/simulation.mp4",
                 use_default_termination: bool = True,
                 reset_options=None):
        super().__init__()
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model and create simulation data.
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.max_time = max_time
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.render_fps = render_fps
        self.renderer = None  # Created on first render call.
        self.viewer = None  # MuJoCo viewer for human rendering.
        self._sim_start_time = None  # Wall-clock time when simulation starts.
        self._frame_count = 0

        # Update metadata with the render fps.
        self.metadata["render_fps"] = self.render_fps

        # Set up camera for rendering.
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.0  # Distance from the robot
        self.camera.elevation = -30  # Camera elevation angle
        self.camera.azimuth = 120  # Camera azimuth angle

        # Set up scene options.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.scene_option.geomgroup[:] = 1  # Enable all geom groups.

        self._get_sensor_idx = lambda name: self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)]
        self._get_vec3_sensor = lambda idx: self.data.sensordata[idx: idx + 3]
        self._get_vec4_sensor = lambda idx: self.data.sensordata[idx: idx + 4]

        # Get sensor indices
        self._body_accel_idx = self._get_sensor_idx("body_accel")
        self._body_gyro_idx = self._get_sensor_idx("body_gyro")
        self._body_vel_idx = self._get_sensor_idx("body_vel")
        self._body_pos_idx = self._get_sensor_idx("body_pos")
        self._body_quat_idx = self._get_sensor_idx("body_quat")
        self._body_linvel_idx = self._get_sensor_idx("body_linvel")
        self._body_xaxis_idx = self._get_sensor_idx("body_xaxis")
        self._body_zaxis_idx = self._get_sensor_idx("body_zaxis")

        # External control inputs (e.g., from a controller).
        self.control_inputs = None

        # Reset options for the environment.
        self.reset_options = reset_options

        # Define the action space: one actuator per joint (12 actuators assumed).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Define observation space using sensor data.
        obs_size = self.model.nsensordata
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Set up modular reward and termination functions.
        self.reward_fns = reward_fns if reward_fns is not None else {"default": self._default_reward}
        self.termination_fns = termination_fns if termination_fns is not None else {}

        # Add default termination function if requested.
        if use_default_termination:
            self.termination_fns["default"] = self._default_termination

        # Video recording settings.
        self.save_video = save_video
        self.video_path = video_path
        self.video_writer = None

        # Seed and initial simulation reset.
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _randomize_initial_state(self):
        """
        Randomly initialize orientation, joint angles, and set respective controls.
        Assumes the standard quadruped structure from quadruped.xml:
        - Free joint (pos + quat)
        - 12 actuated joints (hip, knee, ankle for each of the 4 legs)
        """

        # --- Orientation and Base Velocity Initialization ---
        # Set initial position (e.g., slightly above the ground)
        # You might want to randomize this within a certain area too.
        self.data.qpos[0:3] = [0, 0, 0.25] # Start higher to avoid initial ground contact issues with random orientation

        # Generate a random unit quaternion for arbitrary orientation
        # Method: Generate 4 random numbers from N(0,1), then normalize.
        random_quat = np.random.randn(4)
        random_quat /= np.linalg.norm(random_quat)
        # Ensure the scalar component (w) is positive if needed, though MuJoCo handles both q and -q as the same rotation.
        # if random_quat[0] < 0:
        #     random_quat *= -1
        self.data.qpos[3:7] = random_quat # Initial orientation (w, x, y, z quaternion)

        # Randomly set initial linear and angular velocities for the base
        self.data.qvel[0:3] = np.random.uniform(-0.1, 0.1, size=(3,)) # Linear velocity
        self.data.qvel[3:6] = np.random.uniform(-0.1, 0.1, size=(3,)) # Angular velocity

        # --- Joint Angle and Control Initialization ---

        # Define joint limits (radians) and control ranges based on quadruped.xml
        # Note: XML ranges are in degrees, converted here to radians.
        joint_limits_rad = {
            # <default class="hip"> range="-45 45"
            'hip':   (np.deg2rad(-45), np.deg2rad(45)),
            # <default class="knee"> range="-45 120"
            'knee':  (np.deg2rad(-45), np.deg2rad(120)),
            # <default class="ankle"> range="-90 90"
            'ankle': (np.deg2rad(-90), np.deg2rad(90))
        }
        control_ranges = {
             # <default class="hip"> ctrlrange="-0.5 0.5"
            'hip':   (-0.5, 0.5),
             # <default class="knee"> ctrlrange="-0.91 0.91"
            'knee':  (-0.91, 0.91),
             # <default class="ankle"> ctrlrange="-1 1"
            'ankle': (-1.0, 1.0)
        }
        # Order of joints corresponds to the XML structure and actuator order
        joint_order = ['hip', 'knee', 'ankle'] * 4 # 4 legs

        num_actuated_joints = self.model.nu # Should be 12
        qpos_start_idx = 7 # qpos index after free joint (3 pos + 4 quat)
        qvel_start_idx = 6 # qvel index after free joint (3 lin_vel + 3 ang_vel)

        random_angles_rad = np.zeros(num_actuated_joints)
        control_values = np.zeros(num_actuated_joints)

        # Iterate through each actuated joint
        for i in range(num_actuated_joints):
            joint_type = joint_order[i]
            min_rad, max_rad = joint_limits_rad[joint_type]
            ctrl_min, ctrl_max = control_ranges[joint_type]

            # 1. Generate random angle within joint limits
            angle_rad = np.random.uniform(min_rad, max_rad)
            random_angles_rad[i] = angle_rad

            # 2. Map the target angle (qpos) to the corresponding control value
            joint_range_rad = max_rad - min_rad
            if abs(joint_range_rad) > 1e-6: # Avoid division by zero
                 norm_pos = (angle_rad - min_rad) / joint_range_rad
            else:
                 norm_pos = 0.5 # Midpoint

            ctrl_range = ctrl_max - ctrl_min
            ctrl_val = ctrl_min + norm_pos * ctrl_range

            # Clip control value
            control_values[i] = np.clip(ctrl_val, ctrl_min, ctrl_max)

        # Set the initial joint positions (qpos)
        self.data.qpos[qpos_start_idx : qpos_start_idx + num_actuated_joints] = random_angles_rad

        # Set the initial joint velocities (qvel)
        self.data.qvel[qvel_start_idx : qvel_start_idx + num_actuated_joints] = np.random.uniform(-0.1, 0.1, size=num_actuated_joints) # Or just zeros

        # Set the initial control signals (ctrl)
        self.data.ctrl[:] = control_values

        # Ensure the simulation state reflects these initial values
        mujoco.mj_forward(self.model, self.data)


    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        Also resets the wall-clock timer for real-time synchronization in human mode.
        """
        # Seed if necessary
        if seed is not None:
            self.seed(seed) # Assuming you have a seed method

        if options is None:
            options = self.reset_options

        # Reset MuJoCo data structures to defaults
        mujoco.mj_resetData(self.model, self.data)

        # Apply the custom random initialization
        if options and options.get("randomize_initial_state", False):
            self._randomize_initial_state()

        # Reset simulation time
        self.data.time = 0.0

        # Apply random external controls if specified (assuming self.control_inputs exists)
        if self.control_inputs is not None and options and options.get("control_inputs"):
            self.control_inputs.sample(options=options["control_inputs"])

        # Reset simulation-time trackers
        self._frame_count = 0

        # Set the wall-clock start time if running in human mode
        if self.render_mode == "human":
            self._sim_start_time = time.time()
            # Open the viewer passively if not already open
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

                    # --- Apply visualization options to the viewer ---
                    # Copy flags from self.scene_option to viewer.opt
                    self.viewer.opt.flags[:] = self.scene_option.flags[:]
                    # Copy frame setting
                    self.viewer.opt.frame = self.scene_option.frame
                    # Copy geom group settings
                    self.viewer.opt.geomgroup[:] = self.scene_option.geomgroup[:]
                    # --- End of applying visualization options ---

                    # Apply camera settings from self.camera to the viewer
                    self.viewer.cam.distance = self.camera.distance
                    self.viewer.cam.elevation = self.camera.elevation
                    self.viewer.cam.azimuth = self.camera.azimuth
                    # Set initial lookat based on current qpos (might be randomized)
                    robot_pos = self.data.qpos[:3]
                    self.viewer.cam.lookat[:] = robot_pos

                except Exception as e:
                    print(f"Could not launch MuJoCo viewer: {e}")
                    self.viewer = None
            elif self.viewer.is_running():
                 # If viewer already running, ensure camera lookat is updated
                 robot_pos = self.data.qpos[:3]
                 self.viewer.cam.lookat[:] = robot_pos
                 self.viewer.sync()


        # Initialize video writer if saving video
        if self.save_video and self.video_writer is None:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.render_fps, (self.width, self.height))

        observation = self._get_obs()
        info = {} # Standard Gymnasium practice

        return observation, info

    def _get_obs(self):
        """Obtain observation from the simulation."""
        return self.data.sensordata.copy()

    def _default_reward(self):
        """Default reward function that returns 0."""
        return 0.0

    def _default_termination(self):
        """Default termination: episode ends if simulation time exceeds max_time."""
        return self.data.time >= self.max_time

    def step(self, action):
        """
        Apply the given action, advance the simulation, and return:
        observation, total_reward, terminated, truncated, info.
        """

        if action is not None:
            # Clip the action to the valid range.
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # --- Store previous control state if needed by reward functions ? ---
        # Note: WalkingQuadrupedEnv now handles this internally before calling super().step()
        # previous_ctrl = np.copy(self.data.ctrl) # Might be needed if base rewards use it

        # Step simulation with frame skipping.
        for _ in range(self.frame_skip):
            if action is not None:
                self.data.ctrl[:] = action
                
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # --- Compute rewards and populate info dictionary ---
        total_reward = 0.0
        reward_components = {} # Dictionary to store all components

        for name, fn in self.reward_fns.items():
            # Assume fn() returns either a dictionary of components or a single scalar value
            components_or_scalar = fn()

            if isinstance(components_or_scalar, dict):
                # If it's a dictionary, merge its items into reward_components
                # and add its values to the total reward
                reward_components.update(components_or_scalar)
                total_reward += sum(components_or_scalar.values())
            elif isinstance(components_or_scalar, (float, int, np.number)):
                # If it's a scalar, use the reward function's name as the key
                reward_components[name] = components_or_scalar
                total_reward += components_or_scalar
            else:
                # Handle unexpected return type if necessary
                print(f"Warning: Reward function '{name}' returned unexpected type {type(components_or_scalar)}")


        # Check termination conditions.
        terminated = any(fn() for fn in self.termination_fns.values())
        # Use default termination for truncation unless other logic is added
        truncated = self._default_termination()

        # --- Create the final info dictionary ---
        info = {"time": self.data.time}
        # Add all calculated reward components directly to the info dict
        info.update(reward_components)

        # Add other info if needed, e.g., from self.info if it holds non-reward info
        # if hasattr(self, 'info') and isinstance(self.info, dict):
        #    info.update(self.info) # Be careful not to overwrite reward components

        return observation, total_reward, terminated, truncated, info

    def render_vector(self, origin, vector, color, scale=0.2, radius=0.005, offset=0.0):
        """
        Renders an arrow from origin along 'vector'.
        """
        # Compute the endpoint.
        origin = origin.copy() + np.array([0, 0, offset])
        endpoint = origin + (vector * scale)

        # Check that there is room in the scene.geoms array.
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return

        # Initialize a new geom at index 'ngeom'
        idx = scn.ngeom

        # Set up the arrow geometry.
        arrow = mujoco.MjvGeom()
        arrow.type = mujoco.mjtGeom.mjGEOM_ARROW1
        arrow.rgba[:] = np.array(color, dtype=np.float32)

        # Initialize with default values; then set up with connector:
        mujoco.mjv_initGeom(scn.geoms[idx], arrow.type, np.zeros(3), np.zeros(3), np.zeros(9), arrow.rgba)
        # Use the helper function to compute position, orientation and size from two endpoints.
        # Here, radius determines the thickness of the arrow.
        mujoco.mjv_connector(scn.geoms[idx], arrow.type, radius, origin, endpoint)
        scn.ngeom += 1

    def render_point(self, position, color, radius=0.01):
        """
        Render a point at the given position.
        """
        # Check that there is room in the scene.geoms array.
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return

        # Initialize a new geom at index 'ngeom'
        idx = scn.ngeom

        # Set up the point geometry.
        point = mujoco.MjvGeom()
        point.type = mujoco.mjtGeom.mjGEOM_SPHERE
        point.rgba[:] = np.array(color, dtype=np.float32)
        point.size[:] = [radius, radius, radius]

        # Initialize with default values; then set up with position:
        mujoco.mjv_initGeom(scn.geoms[idx], point.type, point.size, position, np.eye(3, 3).reshape(9) , point.rgba)
        scn.ngeom += 1

    def render_custom_geoms(self):
        """
        Handler for rendering custom geometry.
        By default, do nothing.
        Derived classes can override this to add their own geoms.
        """
        pass

    def update_camera(self):
        """Update the camera to follow the robot."""
        # Get the robot's position
        robot_pos = self.data.qpos[:3]

        # Set the camera position and orientation
        self.camera.lookat[:] = robot_pos

        """
        if self.viewer is not None:
            self.viewer.cam.lookat[0] = self.camera.lookat[0]
            self.viewer.cam.lookat[1] = self.camera.lookat[1]
            self.viewer.cam.lookat[2] = self.camera.lookat[2]
        """
    
    def render(self):
        """
        Render the simulation only when needed based on simulation time and the desired frame rate.

        Behavior for all modes:
          - A new frame is rendered only if simulation time has advanced by at least (1/render_fps) seconds.
          - The frame is then processed (saved to video if enabled and/or returned as an array).
          - In "human" mode, the MuJoCo viewer is used for visualization.
        """
        if self.render_mode is None:
            return None

        # Calculate the expected number of frames based on simulation time and render_fps.
        expected_frames = int(self.data.time * self.render_fps)
        if self._frame_count >= expected_frames:
            return None
        else:
            self._frame_count += 1

        # Create renderer if not already created.
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)

        # Update the camera to follow the robot
        self.update_camera()

        # Update scene and clear any previous custom geoms
        self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=self.camera)

        # Call the handler for custom geometry; if not overridden, nothing happens
        self.render_custom_geoms()

        # Render the frame and convert to BGR format for OpenCV.
        pixels = self.renderer.render()
        
        # Save frame to video if enabled.
        if self.save_video and self.video_writer is not None:
            pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self.video_writer.write(pixels_bgr)

        # Mode-specific handling.
        if self.render_mode == "rgb_array":
            return pixels

        elif self.render_mode == "human":
            # Check if viewer exists and is running before syncing
            if self.viewer is not None and self.viewer.is_running():
                # Wait until wall-clock time catches up to simulation time.
                if self._sim_start_time is None:
                    self._sim_start_time = time.time()

                desired_wall_time = self._sim_start_time + self.data.time
                current_wall_time = time.time()
                wait_time = desired_wall_time - current_wall_time

                if wait_time > 0:
                    time.sleep(wait_time)

                # Sync the viewer with the simulation.
                try: # Add a try-except around sync as well, just in case
                    self.viewer.sync()
                except Exception as e:
                    print(f"Error during viewer sync: {e}")
                    # Optionally close the viewer or handle the error
                    if self.viewer:
                        self.viewer.close()
                    self.viewer = None # Mark viewer as unusable

                # Check again if the viewer was closed by the user during sync/wait
                if self.viewer is not None and not self.viewer.is_running():
                    print("Viewer stopped by the user")
                    self.viewer.close()
                    self.viewer = None
            elif self.viewer is not None and not self.viewer.is_running():
                 # Handle case where viewer was previously initialized but is now closed
                 print("Viewer is closed.")
                 self.viewer.close() # Ensure cleanup
                 self.viewer = None
            # else: viewer is None (failed to initialize or already cleaned up)

            return None # Always return None for human mode
        
    def set_external_control(self, control_inputs: BaseControls):
        """Set external control inputs."""
        self.control_inputs = control_inputs
        

    def close(self):
        """Clean up resources such as the renderer, video writer, mujoco viewer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

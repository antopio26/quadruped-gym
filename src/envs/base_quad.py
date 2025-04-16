# src/envs/base_quad.py
import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import mujoco.viewer
from typing import Optional, Dict, Callable, Any, Tuple, List

class QuadrupedEnv(gym.Env):
    """
    Core Gymnasium environment for a MuJoCo-based quadruped simulation.

    Provides the fundamental interface for interacting with the MuJoCo physics
    engine, including stepping the simulation, accessing sensor data via
    public methods, and handling rendering. Task-specific logic like rewards,
    complex observations, or external control management should be added
    via Gymnasium wrappers.

    Attributes:
        model (mujoco.MjModel): The loaded MuJoCo model.
        data (mujoco.MjData): The MuJoCo simulation data instance.
        max_time (float): Maximum duration of an episode in simulation seconds.
        frame_skip (int): Number of MuJoCo physics steps per environment `step()`.
        render_mode (Optional[str]): Rendering mode ('human', 'rgb_array', or None).
        width (int): Width of the rendered frame for 'rgb_array' mode.
        height (int): Height of the rendered frame for 'rgb_array' mode.
        render_fps (int): Target frames per second for rendering.
        observation_space (gym.spaces.Box): Defines the observation space (defaults to full sensor data).
        action_space (gym.spaces.Box): Defines the action space (based on model actuators).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 model_path: str = "./models/quadruped/scene.xml",
                 max_time: float = 10.0,
                 frame_skip: int = 16,
                 render_mode: Optional[str] = None,
                 width: int = 720,
                 height: int = 480,
                 render_fps: int = 30,
                 save_video: bool = False,
                 video_path: str = "videos/simulation.mp4",
                 reset_options: Optional[Dict[str, Any]] = None):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path

        # Load MuJoCo model and data
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}") from e

        self.max_time = max_time
        self.frame_skip = frame_skip
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1.")

        # Rendering setup
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.render_fps = render_fps
        
        # Update metadata with the render fps.
        self.metadata["render_fps"] = self.render_fps
        
        self.renderer: Optional[mujoco.Renderer] = None
        self.viewer: Optional[mujoco.viewer.Handle] = None
        
        self._sim_start_time: Optional[float] = None
        self._frame_count: int = 0

        # Camera and Scene options
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.0  # Distance from the robot
        self.camera.elevation = -30  # Camera elevation angle
        self.camera.azimuth = 120  # Camera azimuth angle

        # Set up scene options.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.scene_option.geomgroup[:] = 1

        # --- Internal Sensor Helpers (Protected) ---
        self._sensor_indices: Dict[str, int] = self._initialize_sensor_indices([
            "body_accel", "body_gyro", "body_vel", "body_pos", "body_quat",
            "body_linvel", "body_xaxis", "body_zaxis"
            # Add names of joint sensors if they exist in your XML, e.g., "joint_pos_sensor"
        ])
        self._get_vec3_sensor = lambda name: self.data.sensordata[self._sensor_indices[name]: self._sensor_indices[name] + 3]
        self._get_vec4_sensor = lambda name: self.data.sensordata[self._sensor_indices[name]: self._sensor_indices[name] + 4]

        # Reset options
        self.reset_options = reset_options if reset_options is not None else {}

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        # Default observation space uses all available sensor data
        obs_size = self.model.nsensordata
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Video recording
        self.save_video = save_video
        self.video_path = video_path
        self.video_writer: Optional[cv2.VideoWriter] = None

        # Ensure np_random is initialized by Gymnasium's Env base class
        # self.seed() is deprecated; use super().reset(seed=...)

    def _initialize_sensor_indices(self, sensor_names: List[str]) -> Dict[str, int]:
        """Helper to get and validate sensor indices."""
        indices = {}
        missing_sensors = []
        for name in sensor_names:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sensor_id == -1:
                    missing_sensors.append(name)
                else:
                    indices[name] = self.model.sensor_adr[sensor_id]
            except KeyError: # Should not happen with mj_name2id, but good practice
                 missing_sensors.append(name)

        if missing_sensors:
            raise ValueError(f"Sensors not found in MuJoCo model '{self.model_path}': {', '.join(missing_sensors)}. "
                             "Please check sensor names in the XML file.")
        return indices

    def _randomize_initial_state(self):
        """Randomly initializes orientation, joint angles, velocities, and controls."""
        # --- Orientation and Base Velocity ---
        self.data.qpos[0:3] = [0, 0, 0.25] # Start slightly above ground
        random_quat = self.np_random.standard_normal(4) # Use Gym's RNG
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

        mujoco.mj_forward(self.model, self.data) # Apply changes

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Handles seeding via np_random

        # Combine instance options with method options (method options take precedence)
        combined_options = self.reset_options.copy()
        if options is not None:
            combined_options.update(options)

        # Reset MuJoCo simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Apply custom random initialization if requested
        if combined_options.get("randomize_initial_state", False):
            self._randomize_initial_state()
        else:
            # Ensure a forward pass even if not randomizing, to set initial sensor values etc.
            mujoco.mj_forward(self.model, self.data)


        # Reset simulation time and rendering state
        self.data.time = 0.0
        self._frame_count = 0
        self._sim_start_time = None

        # --- Viewer and Video Initialization ---
        if self.render_mode == "human":
            self._sim_start_time = time.time()
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
                    print(f"Warning: Could not launch MuJoCo viewer: {e}")
                    self.viewer = None
            elif self.viewer.is_running():
                 self.viewer.cam.lookat[:] = self.get_body_position() # Use accessor
                 self.viewer.sync()


        # Initialize video writer if saving video
        if self.save_video and self.video_writer is None:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.render_fps, (self.width, self.height))

        observation = self._get_obs()
        info = self._get_info() # Get initial info (usually empty)

        return observation, info

    # --- Public Sensor/State Accessor Methods ---

    def get_body_linear_acceleration(self) -> np.ndarray:
        """Returns the body linear acceleration (IMU) in the global frame."""
        return self._get_vec3_sensor("body_accel").copy()

    def get_body_angular_velocity(self) -> np.ndarray:
        """Returns the body angular velocity (Gyroscope) in the global frame."""
        return self._get_vec3_sensor("body_gyro").copy()

    def get_body_linear_velocity(self) -> np.ndarray:
        """Returns the body linear velocity in the global frame."""
        return self._get_vec3_sensor("body_vel").copy()

    def get_body_position(self) -> np.ndarray:
        """Returns the global position of the body."""
        return self._get_vec3_sensor("body_pos").copy()

    def get_body_orientation_quat(self) -> np.ndarray:
        """Returns the global orientation of the body as a quaternion (w, x, y, z)."""
        return self._get_vec4_sensor("body_quat").copy()

    def get_body_x_axis(self) -> np.ndarray:
        """Returns the body's local X-axis vector in the global frame."""
        return self._get_vec3_sensor("body_xaxis").copy()

    def get_body_z_axis(self) -> np.ndarray:
        """Returns the body's local Z-axis vector (up direction) in the global frame."""
        return self._get_vec3_sensor("body_zaxis").copy()

    def get_actuator_forces(self) -> np.ndarray:
        """Returns the forces/torques applied by the actuators."""
        return self.data.actuator_force.copy()

    def get_control_inputs(self) -> np.ndarray:
        """Returns the current control signals applied to the actuators."""
        return self.data.ctrl.copy()

    def get_joint_angles(self) -> np.ndarray:
        """Returns the current angles of the actuated joints (from qpos)."""
        qpos_start_idx = 7 # Assumes 3 pos + 4 quat for free joint
        num_actuated_joints = self.model.nu
        return self.data.qpos[qpos_start_idx : qpos_start_idx + num_actuated_joints].copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Returns the current velocities of the actuated joints (from qvel)."""
        qvel_start_idx = 6 # Assumes 3 lin_vel + 3 ang_vel for free joint
        num_actuated_joints = self.model.nu
        return self.data.qvel[qvel_start_idx : qvel_start_idx + num_actuated_joints].copy()

    def get_time(self) -> float:
        """Returns the current simulation time."""
        return self.data.time

    def get_dt(self) -> float:
        """Returns the environment step duration (model timestep * frame_skip)."""
        return self.model.opt.timestep * self.frame_skip

    # --- Core Gym Methods ---

    def _get_obs(self) -> np.ndarray:
        """Default observation: returns all available sensor data."""
        return self.data.sensordata.copy()

    def _get_info(self) -> Dict[str, Any]:
        """Returns basic information about the environment state."""
        # Base environment provides minimal info. Wrappers add more.
        return {"time": self.get_time()}

    def _is_terminated(self) -> bool:
        """Checks if the episode should terminate (e.g., robot fell)."""
        # Base environment has no specific termination conditions. Wrappers add these.
        return False

    def _is_truncated(self) -> bool:
        """Checks if the episode should be truncated (e.g., time limit reached)."""
        return self.get_time() >= self.max_time

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies action, steps simulation, returns results."""
        if action is not None:
            # Clip the action to the valid range.
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action and step physics
        try:
            for _ in range(self.frame_skip):
                if action is not None:
                    self.data.ctrl[:] = action
                mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError as e:
            print(f"MuJoCo simulation error: {e}. Resetting environment.")
            # A robust way to handle this might be to return a specific state
            # indicating failure, or reset and return the initial state.
            # For simplicity, we'll return the current (likely invalid) state
            # but mark as terminated.
            observation = self._get_obs() # May contain NaNs or Infs
            reward = 0.0 # Or a large penalty
            terminated = True
            truncated = False # Not truncated due to time limit
            info = self._get_info()
            info["mujoco_error"] = str(e)
            return observation, reward, terminated, truncated, info


        # Get observation, termination, truncation, and info
        observation = self._get_obs()
        terminated = self._is_terminated() # Base env always returns False
        truncated = self._is_truncated()
        info = self._get_info()

        # Base environment provides zero reward. Wrappers add reward calculation.
        reward = 0.0

        # Render if necessary
        self.render()

        return observation, reward, terminated, truncated, info

    # --- Rendering Methods ---

    def render_vector(self, origin: np.ndarray, vector: np.ndarray, color: List[float], scale: float = 0.2, radius: float = 0.005, offset: float = 0.0):
        """Helper to render an arrow geometry in the scene."""
        if self.renderer is None or self.renderer.scene is None: return
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom: return # Check geom buffer space

        origin_offset = origin.copy() + np.array([0, 0, offset])
        endpoint = origin_offset + (vector * scale)
        idx = scn.ngeom
        try:
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, np.zeros(3), np.zeros(3), np.zeros(9), np.array(color, dtype=np.float32))
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, radius, origin_offset, endpoint)
            scn.ngeom += 1
        except IndexError:
             print("Warning: Ran out of geoms in MuJoCo scene for rendering vector.")


    def render_point(self, position: np.ndarray, color: List[float], radius: float = 0.01):
        """Helper to render a sphere geometry at a point."""
        if self.renderer is None or self.renderer.scene is None: return
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom: return

        idx = scn.ngeom
        size = np.array([radius, radius, radius])
        rgba = np.array(color, dtype=np.float32)
        try:
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE, size, position.astype(np.float64), np.eye(3).flatten(), rgba)
            scn.ngeom += 1
        except IndexError:
             print("Warning: Ran out of geoms in MuJoCo scene for rendering point.")

    def render_custom_geoms(self):
        """Placeholder for wrappers to add custom visualizations during rendering."""
        pass # Wrappers should override this if they need custom rendering

    def update_camera(self):
        """Updates the camera lookat point to follow the robot's base."""
        robot_pos = self.get_body_position() # Use public accessor
        self.camera.lookat[:] = robot_pos

        # Update viewer camera if in human mode (not for now)
        if self.render_mode == "human" and self.viewer is not None and self.viewer.is_running():
            # self.viewer.cam.lookat[:] = robot_pos
            pass

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment based on the render_mode."""
        if self.render_mode is None:
            return None

        # Throttle rendering based on render_fps
        sim_time = self.get_time()
        expected_frames = int(sim_time * self.render_fps)
        if self._frame_count >= expected_frames:
            return None # Skip frame
        self._frame_count += 1

        # Initialize renderer if needed
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
            except Exception as e:
                 print(f"Warning: Failed to initialize MuJoCo renderer: {e}")
                 self.render_mode = None # Disable rendering
                 return None

        # Update camera and scene
        self.update_camera()
        try:
            self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=self.camera)
        except mujoco.FatalError as e:
             print(f"Warning: MuJoCo error during scene update: {e}")
             return None # Skip rendering this frame

        # Allow wrappers to add custom geoms
        self.render_custom_geoms()

        # Get pixel data
        try:
            pixels = self.renderer.render()
        except mujoco.FatalError as e:
             print(f"Warning: MuJoCo error during rendering: {e}")
             return None # Skip rendering this frame


        # Save to video if enabled
        if self.save_video and self.video_writer is not None:
            pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self.video_writer.write(pixels_bgr)

        # Handle different render modes
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

    def close(self):
        """Cleans up resources like renderer, viewer, and video writer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.viewer is not None:
            try:
                # Check if running before closing, viewer might crash otherwise
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception as e:
                print(f"Warning: Error closing MuJoCo viewer: {e}")
            self.viewer = None


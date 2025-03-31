import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2

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
                 frame_skip: int = 4,
                 render_mode: str = None,
                 width: int = 720,
                 height: int = 480,
                 render_fps: int = 30,
                 reward_fns: dict = None,
                 termination_fns: dict = None,
                 save_video: bool = False,
                 video_path: str = "videos/simulation.mp4",
                 use_default_termination: bool = True):
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
        self._sim_start_time = None  # Wall-clock time when simulation starts.
        # Internal simulation-time trackers for rendering/video.
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

        # Define the action space: one actuator per joint (12 actuators assumed).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Define observation space using sensor data.
        obs_size = self.model.nsensordata
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Set up modular reward and termination functions.
        self.reward_fns = reward_fns if reward_fns is not None else {"default": self._default_reward}
        self.termination_fns = termination_fns if termination_fns is not None else {}
        if use_default_termination:
            self.termination_fns["default"] = self._default_termination

        # Video recording settings.
        self.save_video = save_video
        self.video_path = video_path
        self.video_writer = None

        # Seed and initial simulation reset.
        self.seed()
        # self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        Also resets the wall-clock timer for real-time synchronization in human mode.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0

        # Set a default control (customize as needed).
        self.data.ctrl[:] = np.array([0, 0, -0.5] * 4)

        # Reset simulation-time trackers.
        self._frame_count = 0

        # Set the wall-clock start time if running in human mode.
        if self.render_mode == "human":
            self._sim_start_time = time.time()

        # Initialize video writer if saving video.
        if self.save_video and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.render_fps, (self.width, self.height))

        observation = self._get_obs()
        return observation, {}

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
        Note: The simulation is stepped as fast as possible; rendering timing is handled in render().
        """
        # Clip the action to the valid range.
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step simulation with frame skipping.
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # Compute rewards using provided reward functions.
        total_reward = 0.0
        reward_info = {}
        for name, fn in self.reward_fns.items():
            r = fn()
            reward_info[name] = r
            total_reward += r

        # Check termination conditions.
        terminated = any(fn() for fn in self.termination_fns.values())
        truncated = False  # Additional truncation conditions can be added if needed.

        info = {"time": self.data.time, "reward_components": reward_info}
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

    def render(self):
        """
        Render the simulation only when needed based on simulation time and the desired frame rate.

        Behavior for all modes:
          - A new frame is rendered only if simulation time has advanced by at least (1/render_fps) seconds.
          - The frame is then processed (saved to video if enabled and/or returned as an array).
          - In "human" mode, the method waits the residual wall-clock time so that the displayed frame
            appears at real-time speed before showing it.
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
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # Save frame to video if enabled.
        if self.save_video and self.video_writer is not None:
            self.video_writer.write(pixels_bgr)

        # Mode-specific handling.
        if self.render_mode == "rgb_array":
            return pixels

        elif self.render_mode == "human":
            # Wait until wall-clock time catches up to simulation time.
            if self._sim_start_time is None:
                self._sim_start_time = time.time()
            desired_wall_time = self._sim_start_time + self.data.time
            current_wall_time = time.time()
            wait_time = desired_wall_time - current_wall_time
            if wait_time > 0:
                time.sleep(wait_time)
            cv2.imshow("Simulation", pixels_bgr)
            cv2.waitKey(1)
            return None

    def close(self):
        """Clean up resources such as the renderer, video writer, and OpenCV windows."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        cv2.destroyAllWindows()

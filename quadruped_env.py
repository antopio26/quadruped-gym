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
    frame-skip, and supports both human and rgb_array render modes with decoupled
    simulation time and rendering FPS.

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
                 termination_fns: dict = None):
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
        self._last_render_time = 0.0

        # Update metadata with the render fps.
        self.metadata["render_fps"] = self.render_fps

        # Set up scene options.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.geomgroup[:] = 1  # Enable all geom groups.

        # Define the action space: one actuator per joint (12 actuators assumed).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Define observation space using sensor data.
        obs_size = self.model.nsensor
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Set up modular reward and termination functions.
        self.reward_fns = reward_fns if reward_fns is not None else {"default": self._default_reward}
        self.termination_fns = termination_fns if termination_fns is not None else {"time": self._default_termination}

        # Seed and initial simulation reset.
        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state and return the initial observation.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0

        # Set a default control (customize as needed).
        self.data.ctrl[:] = np.array([0, 0, -0.5] * 4)
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        """
        Obtain observation from the simulation.
        """
        return self.data.sensordata.copy()

    def _default_reward(self):
        """
        Default reward function that returns 0.
        """
        return 0.0

    def _default_termination(self):
        """
        Default termination: episode ends if simulation time exceeds max_time.
        """
        return self.data.time >= self.max_time

    def step(self, action):
        """
        Apply the given action, advance the simulation, and return:
        observation, total_reward, terminated, truncated, info.
        """
        # Clip the action to the valid range.
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step simulation with frame skipping.
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # Compute rewards using all provided reward functions.
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

    def render(self):
        """
        Render the simulation.
        - In 'rgb_array' mode, returns an image.
        - In 'human' mode, displays the image using OpenCV at the specified render_fps.
        - When render_mode is None, this function returns immediately (for optimal training performance).
        """
        if self.render_mode is None:
            return

        # Create the renderer on first call.
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)

        # Update scene and render frame.
        self.renderer.update_scene(self.data, scene_option=self.scene_option)
        pixels = self.renderer.render()
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        if self.render_mode == "rgb_array":
            return pixels
        elif self.render_mode == "human":
            # Throttle the display to the target FPS.
            current_time = time.time()
            interval = 1.0 / self.render_fps
            if current_time - self._last_render_time >= interval:
                cv2.imshow("Simulation", pixels_bgr)
                cv2.waitKey(1)
                self._last_render_time = current_time
            return

    def close(self):
        """
        Clean up resources such as the renderer and OpenCV windows.
        """
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    # You can pass custom reward/termination functions via dictionaries.
    # Here, we use the defaults.
    env = QuadrupedEnv(render_mode="human", render_fps=10)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()  # Replace with your policy.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # Rendering is decoupled from simulation time.
        done = terminated or truncated

    print("Episode finished with reward:", total_reward)
    env.close()

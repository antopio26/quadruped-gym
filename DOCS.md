# QuadrupedEnv Documentation

The **QuadrupedEnv** class is a custom Gymnasium environment that wraps a MuJoCo simulation of a quadruped. It is designed to be modular so you can easily customize reward components and termination conditions. In addition, the simulation’s physics runs at its natural (time-step) speed while rendering is throttled to a specified frame rate (FPS). When rendering is disabled (by setting `render_mode=None`), the environment incurs no rendering overhead—ideal for fast training.

## Class Overview

Here is the updated documentation with the new parameter `use_default_termination`:

### Key Attributes:

- **Simulation Settings:**
  - `model`: The MuJoCo model loaded from an XML file.
  - `data`: The simulation data object.
  - `max_time`: Maximum episode duration (seconds).
  - `frame_skip`: Number of simulation steps per call to `step()`.
  
- **Rendering Settings:**
  - `render_mode`: `"human"`, `"rgb_array"`, or `None`.  
    - *Human mode* uses OpenCV to display the simulation.
    - *RGB array mode* returns an image as a NumPy array.
    - `None` disables rendering for optimal training performance.
  - `width` and `height`: Dimensions for rendering.
  - `render_fps`: The target FPS for display updates.
  - `save_video`: Whether to save a video of the simulation. Default is `False`.
  - `video_path`: Path to save the video file. Default is `"simulation.mp4"`.
  
- **Modular Functions:**
  - `reward_fns`: A dictionary mapping names (strings) to reward callables. Each callable takes no arguments and returns a numeric reward value.
  - `termination_fns`: A dictionary mapping names to callables that return a Boolean indicating whether the episode should terminate.
  - `use_default_termination`: Whether to use the default termination function. Default is `True`.
  - 
## Using the Environment

### Instantiation

To create an instance of the environment, simply pass the desired parameters. For example:

```python
env = QuadrupedEnv(render_mode="human", render_fps=60)
```

This creates an environment that will display the simulation at 60 FPS while running the simulation at its native time step. For training (when you do not need visualization), set `render_mode=None`:

```python
env = QuadrupedEnv(render_mode=None)
```

### Modular Reward Functions

The environment allows you to combine several reward components. Reward functions are stored in the `reward_fns` dictionary, and at every step the environment calls each function (with no arguments) and sums their outputs to compute the total reward.

**How to Define Reward Functions:**

Reward functions should be defined as callables that access the environment’s internal state (typically through `env.data`). There are two common ways to do this:

1. **Using Lambdas that Capture the Environment:**  
   After instantiating the environment, assign your reward functions to `env.reward_fns`:

   ```python
   def forward_reward(env):
       # Assume forward velocity is stored in qvel[0]
       return env.data.qvel[0]

   def control_cost(env):
       # Penalize high control efforts (using a coefficient of 0.1)
       return -0.1 * np.sum(np.square(env.data.ctrl))

   def alive_bonus(env):
       # Constant bonus for remaining "alive"
       return 1.0

   # Instantiate the environment.
   env = QuadrupedEnv(render_mode="human", render_fps=60)
   # Assign reward functions.
   env.reward_fns = {
       "forward": lambda: forward_reward(env),
       "control_cost": lambda: control_cost(env),
       "alive_bonus": lambda: alive_bonus(env)
   }
   ```
   
**How Rewards Are Combined:**  
At each call to `step()`, the environment iterates over `reward_fns` and sums the returned values. This modular design makes it easy to add, remove, or adjust reward components without modifying the core simulation loop.


2. **Subclassing QuadrupedEnv to Include Reward Methods:**  
   You can subclass the environment and define reward functions as instance methods. This approach is useful when you have many reward components and want to keep the code organized:

   ```python
   class CustomQuadrupedEnv(QuadrupedEnv):
       def forward_reward(self):
           return self.data.qvel[0]

       def control_cost(self):
           return -0.1 * np.sum(np.square(self.data.ctrl))

       def alive_bonus(self):
           return 1.0
   
       def _default_reward(self):
           return self.forward_reward() + self.control_cost() + self.alive_bonus()

   # Instantiate and assign rewards using bound methods.
   env = CustomQuadrupedEnv(render_mode="human", render_fps=60)
   ```

### Modular Termination Conditions

The termination conditions work similarly to reward functions. Each callable in the `termination_fns` dictionary should return a Boolean indicating whether its respective condition is met. The episode terminates if **any** termination condition returns `True`.

**Example Termination Function:**

```python
def fall_termination(env):
    # Terminate if the quadruped's height falls below 0.2.
    # Here we assume env.data.qpos[2] corresponds to the body height.
    return env.data.qpos[2] < 0.2

# Add this condition to the environment.
env.termination_fns["fall"] = lambda: fall_termination(env)
```

### Video Saving Option

The `QuadrupedEnv` class supports saving a video of the simulation. This can be useful for visualizing the agent's behavior during training or evaluation.

**Parameters:**
- `save_video` (bool): Whether to save a video of the simulation. Default is `False`.
- `video_path` (str): Path to save the video file. Default is `"simulation.mp4"`.

**Example Usage:**

To enable video saving, set the `save_video` parameter to `True` and specify the `video_path` if needed:

```python
env = QuadrupedEnv(render_mode="human", render_fps=30, save_video=True, video_path="output/simulation.mp4")
```

This will save the video to the specified path. The video will be recorded at the specified `render_fps`.

**Note:** Ensure that the `cv2` library is installed and properly configured to use the video saving feature.

### Complete Example

Below is a full example that demonstrates how to instantiate the environment, assign custom reward and termination functions, and run a simulation loop:

```python
import numpy as np
from custom_quadruped_env import QuadrupedEnv  # assuming the class is saved in this module

# Define common reward functions.
def forward_reward(env):
    # Reward based on forward velocity (assumes qvel[0] is forward velocity).
    return env.data.qvel[0]

def control_cost(env):
    # Penalize large control inputs.
    return -0.1 * np.sum(np.square(env.data.ctrl))

def alive_bonus(env):
    # Constant bonus for remaining "alive".
    return 1.0

# Define a termination function.
def fall_termination(env):
    # Terminate if the quadruped's body height (qpos[2]) is too low.
    return env.data.qpos[2] < 0.2

# Instantiate the environment.
env = QuadrupedEnv(render_mode="human", render_fps=60)

# Set up reward functions.
env.reward_fns = {
    "forward": lambda: forward_reward(env),
    "control_cost": lambda: control_cost(env),
    "alive_bonus": lambda: alive_bonus(env)
}

# Add a termination condition.
env.termination_fns["fall"] = lambda: fall_termination(env)

# Reset the environment.
obs, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    # Sample a random action (replace with your policy).
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Rendering runs at the specified FPS (if render_mode is not None)
    done = terminated or truncated

print("Episode finished with reward:", total_reward)
env.close()
```

## Class Overview

### QuadrupedEnv

**Attributes:**
- **Simulation Settings:**
  - `model`: The MuJoCo model loaded from an XML file.
  - `data`: The simulation data object.
  - `max_time`: Maximum episode duration (seconds).
  - `frame_skip`: Number of simulation steps per call to `step()`.
  
- **Rendering Settings:**
  - `render_mode`: `"human"`, `"rgb_array"`, or `None`.  
    - *Human mode* uses OpenCV to display the simulation.
    - *RGB array mode* returns an image as a NumPy array.
    - `None` disables rendering for optimal training performance.
  - `width` and `height`: Dimensions for rendering.
  - `render_fps`: The target FPS for display updates.
  - `save_video`: Whether to save a video of the simulation. Default is `False`.
  - `video_path`: Path to save the video file. Default is `"simulation.mp4"`.
  
- **Modular Functions:**
  - `reward_fns`: A dictionary mapping names (strings) to reward callables. Each callable takes no arguments and returns a numeric reward value.
  - `termination_fns`: A dictionary mapping names to callables that return a Boolean indicating whether the episode should terminate.
  - `use_default_termination`: Whether to use the default termination function. Default is `True`.

**Methods:**
- `__init__(self, model_path, max_time, frame_skip, render_mode, width, height, render_fps, reward_fns, termination_fns, save_video, video_path, use_default_termination)`: Initializes the environment with the specified parameters.
- `seed(self, seed=None)`: Sets the random seed for the environment.
- `reset(self, seed=None, options=None)`: Resets the simulation to an initial state and returns the initial observation.
- `_get_obs(self)`: Obtains the observation from the simulation.
- `_default_reward(self)`: Default reward function that returns 0.
- `_default_termination(self)`: Default termination function that ends the episode if the simulation time exceeds `max_time`.
- `step(self, action)`: Applies the given action, advances the simulation, and returns the observation, total reward, termination status, truncation status, and additional info.
- `render_vector(self, origin, vector, color, scale, radius, offset)`: Renders an arrow from the origin along the vector.
- `render_custom_geoms(self)`: Handler for rendering custom geometry. By default, does nothing.
- `render(self)`: Renders the simulation based on the specified render mode.
- `close(self)`: Cleans up resources such as the renderer, video writer, and OpenCV windows.

### ControlInputs

**Attributes:**
- `velocity`: A 3D vector representing the velocity.
- `heading`: A 3D vector representing the heading.

**Methods:**
- `__init__(self)`: Initializes the control inputs with zero vectors for velocity and heading.
- `set_velocity_xy(self, x, y)`: Sets the velocity in the XY plane.
- `set_velocity_speed_alpha(self, speed, alpha)`: Sets the velocity based on speed and angle.
- `set_orientation(self, theta)`: Sets the orientation based on the angle.
- `sample(self, max_speed)`: Randomly samples velocity and orientation.

### WalkingQuadrupedEnv

**Attributes:**
- Inherits all attributes from `QuadrupedEnv`.
- Additional sensor indices for body accelerometer, gyroscope, position, linear velocity, and x-axis.

**Methods:**
- `__init__(self, **kwargs)`: Initializes the environment with additional sensor indices.
- `control_cost(self)`: Penalizes high control inputs.
- `alive_bonus(self)`: Provides a constant bonus for staying "alive".
- `progress_reward(self)`: Rewards moving in the right direction based on velocity.
- `orientation_reward(self)`: Rewards facing the right direction based on heading.
- `_default_reward(self)`: Combines various reward components.
- `render_custom_geoms(self)`: Renders the control inputs as vectors in the simulation.

## Quadruped Robot Model

### Rendering Settings
- **Compiler**: Angle in degrees, mesh directory `./mesh`, texture directory `./textures`
- **Integrator**: Implicit fast

### Default Settings
- **General**:
  - Geom: Mesh type, material `robot_material`, friction `0.6`, margin `0.001`
  - Joint: Axis `0 0 1`, hinge type, damping `0.2`, armature `0.001`
  - Position: Control range `-1 1`, force range `-1.71 1.71`, kp `100`, kv `1`, time constant `0.01`, control limited, force limited

- **Specific Classes**:
  - **Hip**: Joint range `-45 45`, reference `-45`, control range `-0.5 0.5`, gear `0.64`
  - **Knee**: Joint range `-45 120`, reference `37.5`, control range `-0.91 0.91`, gear `0.64`
  - **Ankle**: Joint range `-90 90`, reference `0`, control range `-1 1`, gear `0.64`
  - **Servo**: Geom mesh `SERVO`, mass `0.056`
  - **Frame**: Geom mesh `FRAME`, mass `0.018`
  - **Fema**: Geom mesh `FEMA`, mass `0.022`
  - **Shin**: Geom mesh `SHIN`, mass `0.013`
  - **Foot**: Geom mesh `FOOT`, mass `0.07`

### World Body
- **Frame**:
  - Position: `0.0 0.0 0.1`
  - Child class: `quadruped`
  - Joint: Free type
  - Geoms: `FRAME`, `hip_servo_1`, `hip_servo_2`, `hip_servo_3`, `hip_servo_4`
  - Site: `FRAME`

- **Legs**:
  - **Leg 1**: 
    - Position: `-0.0336 0.02700 0.0195`
    - Joints: `hip_1`, `knee_1`, `ankle_1`
    - Geoms: `fema_1`, `knee_servo_1`, `shin_1`, `foot_1`, `ankle_servo_1`
  - **Leg 2**: 
    - Position: `-0.02700 -0.0336 0.0195`
    - Joints: `hip_2`, `knee_2`, `ankle_2`
    - Geoms: `fema_2`, `knee_servo_2`, `shin_2`, `foot_2`, `ankle_servo_2`
  - **Leg 3**: 
    - Position: `0.0336 -0.02700 0.0195`
    - Joints: `hip_3`, `knee_3`, `ankle_3`
    - Geoms: `fema_3`, `knee_servo_3`, `shin_3`, `foot_3`, `ankle_servo_3`
  - **Leg 4**: 
    - Position: `0.0270 0.0336 0.0195`
    - Joints: `hip_4`, `knee_4`, `ankle_4`
    - Geoms: `fema_4`, `knee_servo_4`, `shin_4`, `foot_4`, `ankle_servo_4`

### Assets
- **Meshes**: `FRAME.obj`, `FEMA.obj`, `SHIN.obj`, `FOOT.obj`, `SERVO.obj`
- **Texture**: `colors.png`
- **Material**: `robot_material`, texture `robot_texture`, texuniform `true`, rgba `1 1 1 1`

### Actuators
| Index | Joint  | Class |
|-------|--------|-------|
| 0     | hip_1  | hip   |
| 1     | knee_1 | knee  |
| 2     | ankle_1| ankle |
| 3     | hip_2  | hip   |
| 4     | knee_2 | knee  |
| 5     | ankle_2| ankle |
| 6     | hip_3  | hip   |
| 7     | knee_3 | knee  |
| 8     | ankle_3| ankle |
| 9     | hip_4  | hip   |
| 10    | knee_4 | knee  |
| 11    | ankle_4| ankle |

### Sensors
| Index | Joint/Site | Sensor Name    |
|-------|------------|----------------|
| 0     | hip_1      | hip_1_sensor   | 
| 1     | knee_1     | knee_1_sensor  |
| 2     | ankle_1    | ankle_1_sensor |
| 3     | hip_2      | hip_2_sensor   |
| 4     | knee_2     | knee_2_sensor  |
| 5     | ankle_2    | ankle_2_sensor |
| 6     | hip_3      | hip_3_sensor   |
| 7     | knee_3     | knee_3_sensor  |
| 8     | ankle_3    | ankle_3_sensor |
| 9     | hip_4      | hip_4_sensor   |
| 10    | knee_4     | knee_4_sensor  |
| 11    | ankle_4    | ankle_4_sensor |
| 12    | FRAME      | body_accel[x]  |
| 13    | FRAME      | body_accel[y]  |
| 14    | FRAME      | body_accel[z]  |
| 15    | FRAME      | body_gyro[x]   |
| 16    | FRAME      | body_gyro[y]   |
| 17    | FRAME      | body_gyro[z]   |
| 18    | FRAME      | body_pos[x]    |
| 19    | FRAME      | body_pos[y]    |
| 20    | FRAME      | body_pos[z]    |
| 21    | FRAME      | body_linvel[x] |
| 22    | FRAME      | body_linvel[y] |
| 23    | FRAME      | body_linvel[z] |
| 24    | FRAME      | body_xaxis[x]  |
| 25    | FRAME      | body_xaxis[y]  |
| 26    | FRAME      | body_xaxis[z]  |
| 27    | FRAME      | body_zaxis[x]  |
| 28    | FRAME      | body_zaxis[y]  |
| 29    | FRAME      | body_zaxis[z]  |

---

# Summary

- **Instantiation:**  
  Create an environment instance with parameters such as `render_mode`, `render_fps`, and `frame_skip`.

- **Reward Functions:**  
  Define reward functions as callables that reference the environment’s internal state. They are stored in the `reward_fns` dictionary, and the total reward is the sum of all individual components.

- **Termination Conditions:**  
  Define termination functions similarly using the `termination_fns` dictionary. An episode ends when any termination condition returns `True`.

- **Combining Rewards:**  
  Multiple reward components can be combined by simply summing their outputs. You can adjust coefficients and add as many components as needed.

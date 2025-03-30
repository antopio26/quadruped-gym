# QuadrupedEnv Documentation

The Quadruped simulation environment is built on the MuJoCo physics engine and is designed for both training and visualization. Its modular design allows you to customize reward components, termination conditions, and rendering settings. The main classes covered include:

- **QuadrupedEnv:** The base environment class.
- **VelocityHeadingControls:** A helper class to manage velocity and heading inputs.
- **WalkingQuadrupedEnv:** An extension of QuadrupedEnv that includes additional sensors and reward components for walking behaviors.

## Environment Configuration

### Simulation Settings
- **model:** The MuJoCo model loaded from an XML file.
- **data:** The simulation data object.
- **max_time:** Maximum episode duration (seconds).
- **frame_skip:** Number of simulation steps per call to `step()`.

### Rendering Settings
- **render_mode:** Options include `"human"`, `"rgb_array"`, or `None`.
  - *Human mode* displays the simulation using OpenCV.
  - *RGB array mode* returns an image as a NumPy array.
  - Setting to `None` disables rendering for optimal training performance.
- **width** and **height:** Dimensions for rendering.
- **render_fps:** Target frames per second for display updates.
- **save_video:** If `True`, the simulation video will be saved.
- **video_path:** File path to save the video (default: `"simulation.mp4"`).

### Modular Functions
- **Reward Functions (`reward_fns`):**
  A dictionary mapping names to reward callables (each takes no arguments and returns a numeric value). These functions are summed at each simulation step.

- **Termination Functions (`termination_fns`):**
  A dictionary mapping names to callables that return a Boolean indicating whether the episode should terminate. The episode ends if any condition is met.

- **use_default_termination:**
  A flag to enable the built-in termination based on simulation time.

## Using the Environment

### Instantiation

Create an environment instance by passing the desired parameters. For example:

```python
env = QuadrupedEnv(render_mode="human", render_fps=60)
```

For training without visualization:

```python
env = QuadrupedEnv(render_mode=None)
```

### Modular Reward Functions

Reward functions can be set up in two ways:

1. **Assigning Callables After Instantiation:**
   Define functions that access `env.data` and then add them to the `reward_fns` dictionary:

   ```python
   def forward_reward(env):
       return env.data.qvel[0]

   def control_cost(env):
       return -0.1 * np.sum(np.square(env.data.ctrl))

   def alive_bonus(env):
       return 1.0

   env = QuadrupedEnv(render_mode="human", render_fps=60)
   env.reward_fns = {
       "forward": lambda: forward_reward(env),
       "control_cost": lambda: control_cost(env),
       "alive_bonus": lambda: alive_bonus(env)
   }
   ```

2. **Subclassing QuadrupedEnv:**
   Define reward functions as instance methods for better organization:

   ```python
   class CustomQuadrupedEnv(QuadrupedEnv):
       def forward_reward(self):
           return self.data.qvel[0]

       def control_cost(self):
           return -0.1 * np.sum(np.square(self.data.ctrl))

       def alive_bonus(self):
           return 1.0

       # Override default reward function
       def _default_reward(self):
           return self.forward_reward() + self.control_cost() + self.alive_bonus()

   env = CustomQuadrupedEnv(render_mode="human", render_fps=60)
   ```

   You can override the default reward function or again use lambdas that wrap the child class methods and setting them as custom reward functions.

   ```python
   class CustomQuadrupedEnv(QuadrupedEnv):
       def forward_reward(self):
           return self.data.qvel[0]

       def control_cost(self):
           return -0.1 * np.sum(np.square(self.data.ctrl))

       def alive_bonus(self):
           return 1.0

       # No override of default reward function

   env = CustomQuadrupedEnv(render_mode="human", render_fps=60)
   env.reward_fns = {
       "forward": lambda: env.forward_reward(),
       "control_cost": lambda: env.control_cost(),
       "alive_bonus": lambda: env.alive_bonus()
   }
   ```

### Modular Termination Conditions

Termination conditions work similarly to rewards. Each callable in the `termination_fns` dictionary should return a Boolean value. For instance:

```python
def fall_termination(env):
    return env.data.qpos[2] < 0.2

env.termination_fns["fall"] = lambda: fall_termination(env)
```

### Video Saving Option

Enable video recording by setting `save_video` to `True` and providing a `video_path` if needed:

```python
env = QuadrupedEnv(render_mode="human", render_fps=30, save_video=True, video_path="output/simulation.mp4")
```

Make sure the `cv2` library is installed and properly configured.

## Custom Geometry Rendering

The `render_custom_geoms` method in the `QuadrupedEnv` class allows you to add custom helper geometry to the scene. This can be useful for visualizing control inputs, forces, or other vectors during the simulation.

### Method Overview

```python
def render_custom_geoms(self):
    """
    Handler for rendering custom geometry.
    By default, do nothing.
    Derived classes can override this to add their own geoms.
    """
    pass
```

### Example Usage

To use `render_custom_geoms`, you need to override it in a subclass and add your custom rendering logic. For example, in the `WalkingQuadrupedEnv` class, this method is used to render control input vectors:

```python
class WalkingQuadrupedEnv(QuadrupedEnv):

    def render_custom_geoms(self):
        # Render the control inputs as vectors.
        origin = self._get_vec3_sensor(self._body_pos_idx)

        # Render the velocity vector in red
        self.render_vector(origin, self.control_inputs.velocity, [1, 0, 0, 1], offset=0.05)
        # Render the heading vector in green
        self.render_vector(origin, self.control_inputs.heading, [0, 1, 0, 1], offset=0.05)
```

### Adding Custom Geometry

1. **Override `render_custom_geoms`**: Create a subclass of `QuadrupedEnv` and override the `render_custom_geoms` method.
2. **Use `render_vector`**: Call the `render_vector` method to add arrows or other shapes to the scene.

### Example

Here is a complete example of how to add custom geometry to the scene:

```python
class CustomQuadrupedEnv(QuadrupedEnv):

    def render_custom_geoms(self):
        # Example: Render a custom vector
        origin = self._get_vec3_sensor(self._body_pos_idx)
        custom_vector = np.array([1, 0, 0])  # Example vector
        self.render_vector(origin, custom_vector, [0, 0, 1, 1], offset=0.1)  # Blue vector
```

In this example, a blue vector is rendered at the position of the quadruped's body. You can customize the vector's direction, color, and offset as needed.

## Example: Complete Simulation Loop

Below is a full example demonstrating instantiation, custom reward and termination functions, and a simulation loop:

```python
import numpy as np
from quadruped_env import QuadrupedEnv

# Define reward functions.
def forward_reward(env):
    return env.data.qvel[0]

def control_cost(env):
    return -0.1 * np.sum(np.square(env.data.ctrl))

def alive_bonus(env):
    return 1.0

# Define a termination function.
def fall_termination(env):
    return env.data.qpos[2] < 0.2

# Instantiate the environment.
env = QuadrupedEnv(render_mode="human", render_fps=60)

# Set reward and termination functions.
env.reward_fns = {
    "forward": lambda: forward_reward(env),
    "control_cost": lambda: control_cost(env),
    "alive_bonus": lambda: alive_bonus(env)
}
env.termination_fns["fall"] = lambda: fall_termination(env)

# Run the simulation loop.
obs, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Render if not disabled
    done = terminated or truncated

print("Episode finished with reward:", total_reward)
env.close()
```

## Class Details

### QuadrupedEnv

**Key Attributes:**
- **Simulation:** `model`, `data`, `max_time`, `frame_skip`.
- **Rendering:** `render_mode`, `width`, `height`, `render_fps`, `save_video`, `video_path`.
- **Modular Functions:** `reward_fns`, `termination_fns`, `use_default_termination`.

**Important Methods:**
- `__init__`: Initializes the environment with the specified parameters.
- `seed`: Sets the random seed.
- `reset`: Resets the simulation to an initial state.
- `_get_obs`: Retrieves the current observation.
- `_default_reward`: Returns a default reward (often 0).
- `_default_termination`: Terminates the episode based on `max_time`.
- `step`: Advances the simulation and returns observation, reward, termination status, and info.
- `render`: Displays the simulation based on the current render mode.
- `close`: Cleans up resources (e.g., OpenCV windows, video writer).

### VelocityHeadingControls

Manages control variables for the simulation.

**Attributes:**
- `velocity`: A 3D vector for velocity.
- `heading`: A 3D vector for heading.

**Methods:**
- `__init__`: Initializes velocity and heading to zero.
- `set_velocity_xy`: Sets the velocity in the XY plane.
- `set_velocity_speed_alpha`: Sets velocity based on speed and angle.
- `set_orientation`: Sets the heading using an angle.
- `sample`: Randomly samples velocity and orientation within limits.

### WalkingQuadrupedEnv

Extends QuadrupedEnv with additional sensor inputs and rewards tailored for walking.

**Additional Attributes:**
- Sensor indices for body accelerometer, gyroscope, position, linear velocity, and x-axis.

**Additional Methods:**
- `control_cost`: Penalizes high control inputs.
- `alive_bonus`: Provides a constant bonus for staying “alive.”
- `progress_reward`: Rewards forward movement.
- `orientation_reward`: Rewards correct facing direction.
- `_default_reward`: Combines multiple reward components.
- `render_custom_geoms`: Renders vectors representing control inputs in the simulation.

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
| 30    | FRAME      | body_vel[x]    |
| 31    | FRAME      | body_vel[y]    |
| 32    | FRAME      | body_vel[z]    |

>> **NOTE**: body_linvel is the velocity vector of the FRAME in global coordinates while body_vel is the velocity of the frame in local coordinates (it acts as optial flow sensor)
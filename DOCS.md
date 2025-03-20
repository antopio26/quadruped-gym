# QuadrupedEnv Documentation

The **QuadrupedEnv** class is a custom Gymnasium environment that wraps a MuJoCo simulation of a quadruped. It is designed to be modular so you can easily customize reward components and termination conditions. In addition, the simulation’s physics runs at its natural (time-step) speed while rendering is throttled to a specified frame rate (FPS). When rendering is disabled (by setting `render_mode=None`), the environment incurs no rendering overhead—ideal for fast training.

## Class Overview

**Key Attributes:**

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
  
- **Modular Functions:**
  - `reward_fns`: A dictionary mapping names (strings) to reward callables. Each callable takes no arguments and returns a numeric reward value.
  - `termination_fns`: A dictionary mapping names to callables that return a Boolean indicating whether the episode should terminate.

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

2. **Subclassing QuadrupedEnv to Include Reward Methods:**  
   You can subclass the environment and define reward functions as instance methods. Then, pass them (via lambdas) when setting up the environment:

   ```python
   class CustomQuadrupedEnv(QuadrupedEnv):
       def forward_reward(self):
           return self.data.qvel[0]

       def control_cost(self):
           return -0.1 * np.sum(np.square(self.data.ctrl))

       def alive_bonus(self):
           return 1.0

   # Instantiate and assign rewards using bound methods.
   env = CustomQuadrupedEnv(render_mode="human", render_fps=60)
   env.reward_fns = {
       "forward": lambda: env.forward_reward(),
       "control_cost": lambda: env.control_cost(),
       "alive_bonus": lambda: env.alive_bonus()
   }
   ```

**How Rewards Are Combined:**  
At each call to `step()`, the environment iterates over `reward_fns` and sums the returned values. This modular design makes it easy to add, remove, or adjust reward components without modifying the core simulation loop.

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

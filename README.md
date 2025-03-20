# Quadruped Simulation Environment

## 3D Printable Model

The robot model used in this simulation mirrors the following 3D printable robot: [MG 996R Quadruped Robot](https://makerworld.com/en/models/904181-mg-996r-quadruped-robot).

## Overview

This project provides a custom Gymnasium environment for simulating a quadruped robot using MuJoCo. The environment supports various sensors, including accelerometers, gyroscopes, and position sensors, and allows for modular reward functions and termination conditions.

## Features

- **Simulation Settings:**
  - **Model:** MuJoCo model loaded from an XML file.
  - **Data:** Simulation data object.
  - **Max Time:** Maximum episode duration (seconds).
  - **Frame Skip:** Number of simulation steps per call to `step()`.

- **Rendering Settings:**
  - **Render Mode:** `"human"`, `"rgb_array"`, or `None`.
    - *Human mode* uses OpenCV to display the simulation.
    - *RGB array mode* returns an image as a NumPy array.
    - `None` disables rendering for optimal training performance.
  - **Width and Height:** Dimensions for rendering.
  - **Render FPS:** The target FPS for display updates.

- **Modular Functions:**
  - **Reward Functions:** A dictionary mapping names to reward callables.
  - **Termination Functions:** A dictionary mapping names to callables that return a Boolean indicating whether the episode should terminate.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/antopio26/quadruped-gym.git
   cd quadruped-simulation
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Ensure you have MuJoCo installed and properly configured.

## Usage

### Instantiation

To create an instance of the environment, pass the desired parameters:

```python
from quadruped_env import QuadrupedEnv

env = QuadrupedEnv(render_mode="human", render_fps=60)
```

### Modular Reward Functions

Define reward functions as callables that access the environment’s internal state. Assign them to `env.reward_fns`:

```python
def forward_reward(env):
    return env.data.qvel[0]

def control_cost(env):
    return -0.1 * np.sum(np.square(env.data.ctrl))

def alive_bonus(env):
    return 1.0

env.reward_fns = {
    "forward": lambda: forward_reward(env),
    "control_cost": lambda: control_cost(env),
    "alive_bonus": lambda: alive_bonus(env)
}
```

### Modular Termination Conditions

Define termination functions and assign them to `env.termination_fns`:

```python
def fall_termination(env):
    return env.data.qpos[2] < 0.2

env.termination_fns["fall"] = lambda: fall_termination(env)
```

### Running the Simulation

Reset the environment and run a simulation loop:

```python
obs, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    done = terminated or truncated

print("Episode finished with reward:", total_reward)
env.close()
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
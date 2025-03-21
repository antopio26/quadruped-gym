import numpy as np
from envs.quadruped import WalkingQuadrupedEnv

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Instantiate the environment.
env = WalkingQuadrupedEnv(render_mode="human", render_fps=30, save_video=True)

# Sample random control inputs.
ctrl_vel, ctrl_heading = env.control_inputs.sample()

print("Control inputs:")
print("Velocity:", ctrl_vel)
print("Heading:", ctrl_heading)

obs, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # Replace with your policy.
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    frame = env.render()  # Render returns a frame (for rgb_array) or displays it (for human).

    done = terminated or truncated

print("Episode finished with reward:", total_reward)
env.close()

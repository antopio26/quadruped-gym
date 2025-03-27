import os
import numpy as np
import pandas as pd
from envs.po_walking_quad import WalkingQuadrupedEnv
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.callbacks import BaseCallback
from utils.plot import plot_data
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Instantiate the environment.
env = WalkingQuadrupedEnv(render_mode="human", render_fps=30, save_video=True, frame_window=3, random_controls=True)

# env.control_inputs.set_velocity_speed_alpha(speed=1, alpha=0)
# env.control_inputs.set_orientation(theta=0)

# Define the model
model = SAC("MlpPolicy", env)

# Callback to record rewards
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals["rewards"])
        return True

reward_callback = RewardCallback()

filepath = "./policies/sac_quadruped.zip"

# If the model file exists, load it
if filepath and os.path.isfile(filepath):
    model = model.load(filepath, env=env)
    print("Previous model loaded.")

for i in range(10):

    # Train the model
    model.learn(total_timesteps=10_000, progress_bar=True, callback=reward_callback)

    # Save the model
    model.save(filepath)

    # Prepare data for plotting
    data = pd.DataFrame({
        'Training Steps': range(len(reward_callback.rewards)),
        'Reward': [reward[0] for reward in reward_callback.rewards],  # Extract the reward value
        'Condition1': ['Training'] * len(reward_callback.rewards)
    })

    # Plot the rewards over training steps using plot_data
    plot_data([data], xaxis='Training Steps', value='Reward', condition='Condition1', bin_size=200)
    plt.show()

    # Evaluate the model
    obs, _ = env.reset()

    # Update video path
    new_video_path = f'./videos/runs/sac_quadruped_{i}.mp4'
    env.update_video_path(new_video_path)

    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()

env.close()
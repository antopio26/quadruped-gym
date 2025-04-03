import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.po_walking_quad import POWalkingQuadrupedEnv
from utils.plot import plot_data_line
import matplotlib.pyplot as plt

# Function to create a new environment instance
def make_env(reset_options=None):
    new_env = POWalkingQuadrupedEnv(
        max_time=20,
        obs_window=5,
        random_controls=True,
        reset_options=reset_options
    )
    return new_env

if __name__ == '__main__':
    output_folder = '../policies/po_v2_ppo_lip_freq_v1'
    os.makedirs(output_folder, exist_ok=True)

    # Create subfolders for logs, videos and plots
    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'plots'), exist_ok=True)

    # Define the options dictionary
    options = {
        # 'min_speed': 0.0,
        # 'max_speed': 0.5,
        'fixed_heading_angle': 0.0,
        'fixed_velocity_angle': 0.0,
        'fixed_speed': 0.3
    }

    # Create a vectorized environment with 10 parallel environments
    num_envs = 10
    env = SubprocVecEnv([lambda: make_env(options) for _ in range(num_envs)])

    # Define the model
    model = PPO("MlpPolicy", env)

    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(RewardCallback, self).__init__(verbose)
            self.rewards = []
            self.std = []

        def _on_step(self) -> bool:
            self.rewards.append(np.mean(self.locals["rewards"]))  # Get rewards from all envs
            self.std.append(np.std(self.locals["rewards"]))  # Get std from all envs
            return True

    reward_callback = RewardCallback()

    filepath = os.path.join(output_folder, 'policy.zip')
    steps_filepath = os.path.join(output_folder, 'steps.txt')

    # If the model file exists, load it
    if filepath and os.path.isfile(filepath):
        model = model.load(filepath, env=env)
        print("Previous model loaded.")
        if os.path.isfile(steps_filepath):
            with open(steps_filepath, 'r') as f:
                start_step = int(f.read())
        else:
            start_step = 0
    else:
        start_step = 0

    # Train the model for n steps
    num_steps = 20

    for i in range(start_step, start_step + num_steps):
        # Train the model
        model.learn(total_timesteps=500_000, progress_bar=True, callback=reward_callback)

        # Save the model
        model.save(filepath)

        # Save the current step
        with open(steps_filepath, 'w') as f:
            f.write(str(i + 1))

        # Prepare data for plotting
        data = pd.DataFrame({
            'Training Steps': range(len(reward_callback.rewards)),
            'Reward': reward_callback.rewards,
            'Std': reward_callback.std,
            'Condition1': 'Training'
        })

        # Save the dataframe
        data.to_csv(os.path.join(output_folder, f'logs/rewards_{i}.csv'), index=False)

        # Plot the rewards over training steps using plot_data
        plot_data_line([data], xaxis='Training Steps', value='Reward', condition='Condition1', smooth=len(reward_callback.rewards)//100)
        plt.savefig(os.path.join(output_folder, f'plots/reward_plot_{i}.png'))
        plt.close()

        # Update video path
        new_video_path = os.path.join(output_folder, f'videos/run_{i}.mp4')

        env = POWalkingQuadrupedEnv(
            max_time=20,
            render_mode="human",
            save_video=True,
            video_path=new_video_path,
            obs_window=5,
            random_controls=True,
            reset_options=options
        )

        # Evaluate the model
        obs, _ = env.reset()

        done = False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            env.render()

        env.close()
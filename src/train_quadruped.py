import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- Add this section ---
import matplotlib
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot
# --- End of added section ---

import matplotlib.pyplot as plt # Keep this import after setting the backend

from src.envs.po_quad import POWalkingQuadrupedEnv
from src.utils.plot import plot_data_line, plot_reward_components
from src.callbacks.reward_callback import RewardCallback

# Function to create a new environment instance
def make_env(reset_options=None):
    new_env = POWalkingQuadrupedEnv(
        max_time=20,
        reset_options=reset_options
    )

    return new_env

if __name__ == '__main__':
    real_time_flag = True
    output_folder = './policies/po_new_ppo_v1'
    os.makedirs(output_folder, exist_ok=True)

    # Create subfolders for logs, videos and plots
    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'plots'), exist_ok=True)

    # Define the options dictionary
    options = {
        'randomize_initial_state': True,
        'control_inputs': {
            'min_speed': 0.0,
            'max_speed': 0.4,
            'fixed_heading_angle': None,
            'fixed_velocity_angle': None,
            'fixed_speed': None
        }
    }

    # Create a vectorized environment with 10 parallel environments
    num_envs = 10
    env = SubprocVecEnv([lambda: make_env(options) for _ in range(num_envs)])

    # Define the model
    model = PPO("MlpPolicy", env) # RecurrentPPO("MlpLstmPolicy", env, device = "cuda")

    # Pass the output_folder for saving the continuous log
    reward_callback = RewardCallback(output_folder=output_folder, real_time_flag=real_time_flag)

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

        steps = range(len(reward_callback.data['rewards']))
        df = pd.DataFrame({
            'Training Steps': steps,
            'Condition': ['Training'] * len(steps),
            'Reward': reward_callback.data['rewards'],
            'Std': reward_callback.data['std']
        })
        
        components_df = pd.DataFrame(reward_callback.data['components'])
        data =  pd.concat([df[['Training Steps', 'Condition']], 
                        components_df, 
                        df[['Reward', 'Std']]], axis=1)[reward_callback.column_order]


        # Save the dataframe
        data.to_csv(os.path.join(output_folder, f'logs/rewards_{i}.csv'), index=False)

        # Plot the rewards over training steps using plot_data
        plot_data_line([data], xaxis='Training Steps', value='Reward', condition='Condition', smooth=len(reward_callback.data['rewards'])//100)
        plt.savefig(os.path.join(output_folder, f'plots/reward_plot_{i}.png'))

        plot_reward_components(data, output=os.path.join(output_folder, f'plots/reward_components_plot.html'))
        plt.close()

        # Update video path
        new_video_path = os.path.join(output_folder, f'videos/run_{i}.mp4')

        env = POWalkingQuadrupedEnv(
            max_time=20,
            render_mode="human",
            save_video=True,
            video_path=new_video_path,
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
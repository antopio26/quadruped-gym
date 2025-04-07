import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.po_walking_quad import POWalkingQuadrupedEnv
from utils.plot import plot_data_line, plot_reward_components
import matplotlib.pyplot as plt
import csv
import torch

# Function to create a new environment instance
def make_env(reset_options=None):
    new_env = POWalkingQuadrupedEnv(
        max_time=20,
        obs_window=5,
        random_controls=True,
        reset_options=reset_options
    )

    # Set the random seed for reproducibility
    new_env.seed(10)

    return new_env

if __name__ == '__main__':
    output_folder = '../policies/po_minimal_omni_ppo_v2'
    os.makedirs(output_folder, exist_ok=True)

    # Create subfolders for logs, videos and plots
    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'plots'), exist_ok=True)

    # Define the options dictionary
    options = {
        # 'min_speed': 0.0,
        # 'max_speed': 0.4,
        'fixed_heading_angle': 0.0,
        'fixed_velocity_angle': None,
        'fixed_speed': 0.3
    }

    # Create a vectorized environment with 10 parallel environments
    num_envs = 10
    env = SubprocVecEnv([lambda: make_env(options) for _ in range(num_envs)])

    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))

    # Define the model
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs) # RecurrentPPO("MlpLstmPolicy", env, device = "cuda")

    class RewardCallback(BaseCallback):
        def __init__(self, keys, real_time_flag = True ,verbose=0):
            super().__init__(verbose)
            self.keys = keys
            self.data = {
                'rewards': [],
                'std': [],
                'components': {key: [] for key in self.keys}
            }
            
            self.column_order = ['Training Steps'] + self.keys + ['Reward', 'Std', 'Condition']

            self.real_time_column = ['Training Steps'] + self.keys + ['Reward']

            self.step_counter = 0
            self.csv_file = os.path.join('./rewards_continuous.csv')

            self.real_time_flag = real_time_flag

            if self.real_time_flag:
                if os.path.exists(self.csv_file):
                    os.remove(self.csv_file)
                
                with open(self.csv_file, 'w') as f:
                    f.write(','.join(self.real_time_column) + '\n')

        def _on_step(self) -> bool:
            infos = self.locals["infos"]
            
            current_components = {
                key: np.mean([info[key] for info in infos]) 
                for key in self.keys
            }
            
            self.data['rewards'].append(np.mean(self.locals["rewards"]))
            self.data['std'].append(np.std(self.locals["rewards"]))
            for key in self.keys:
                self.data['components'][key].append(current_components[key])            

            if self.real_time_flag:
                row_data = {
                    'Training Steps': self.step_counter,
                    'Reward': self.data['rewards'][-1],
                }
                row_data.update(current_components)
                self.step_counter += 1

                with open(self.csv_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=self.real_time_column)
                    writer.writerow(row_data)
            return True
    
    reward_callback = RewardCallback(POWalkingQuadrupedEnv.reward_keys, real_time_flag=real_time_flag)

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
    num_steps = 50

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
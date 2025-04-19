import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO # Or SAC, TD3 etc.
# from sb3_contrib import RecurrentPPO # Uncomment if using LSTM
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

# Matplotlib backend setting (important for non-GUI servers)
import matplotlib
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot
import matplotlib.pyplot as plt

# Import the environment builder and control logic
from src.envs.env_builder import create_quadruped_env
from src.controls.velocity_heading_controls import VelocityHeadingControls

# Import utilities and callbacks
from src.utils.plot import plot_data_line, plot_reward_components
from src.callbacks.reward_callback import RewardCallback

# --- Environment Creation Function for Vectorized Env ---
def make_env(rank: int, seed: int = 0, env_options: dict = None):
    """
    Utility function for multiprocessed env.

    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :param env_options: Dictionary of options for create_quadruped_env
    """
    if env_options is None:
        env_options = {}
    def _init():
        # Ensure each subprocess gets a different seed if desired
        # np.random.seed(seed + rank) # Seeding handled by SB3 VecEnv wrapper usually
        env = create_quadruped_env(**env_options)
        # env.seed(seed + rank) # Deprecated, use reset(seed=...)
        # env.reset(seed=seed+rank) # SB3 VecEnv handles seeding
        return env
    # set_global_seeds(seed) # Deprecated
    return _init

if __name__ == '__main__':
    # --- Configuration ---
    REAL_TIME_PLOT = False # Set to True for live plotting (can be slow)
    OUTPUT_FOLDER = './policies/po_new_ppo_v2_wrapped' # Choose a new folder name
    MODEL_FILENAME = 'policy.zip'
    STEPS_FILENAME = 'steps.txt'
    LOGS_SUBDIR = 'logs'
    VIDEOS_SUBDIR = 'videos'
    PLOTS_SUBDIR = 'plots'

    # Environment parameters
    OBS_WINDOW = 1 # Observation window size (1 for single timestep, >1 for history)
    MAX_TIME = 20.0 # Max seconds per episode
    NUM_ENVS = 10 # Number of parallel environments for training
    CONTROL_LOGIC = VelocityHeadingControls

    # Training parameters
    MODEL_CLASS = PPO # PPO, SAC, TD3, RecurrentPPO
    POLICY = "MlpPolicy" # "MlpPolicy" or "MlpLstmPolicy" for RecurrentPPO
    TOTAL_TIMESTEPS_PER_LEARN = 500_000 # Timesteps per call to model.learn()
    NUM_LEARN_CALLS = 20 # Total training = TOTAL_TIMESTEPS_PER_LEARN * NUM_LEARN_CALLS
    LEARN_KWARGS = {"progress_bar": True} # Add other SB3 learn kwargs if needed
    VERBOSE = 0 # Verbosity level for SB3 (0=none, 1=info, 2=debug)

    # Reset options for training (passed to ControlInputWrapper.sample)
    TRAIN_RESET_OPTIONS = {
        'randomize_initial_state': True, # Randomize base pose/velocity in BaseQuadrupedEnv
        'control_inputs': {             # Options for ControlInputWrapper.sample
            'min_speed': 0.0,
            'max_speed': 0.4,
            'fixed_heading_angle': None,    # Random heading
            'fixed_velocity_angle': None,   # Random local velocity angle
            'fixed_speed': None             # Random speed
        }
        # Add other BaseQuadrupedEnv reset options if needed
    }

    # Evaluation parameters (for saving video)
    EVAL_MAX_TIME = 20.0 # Longer time for evaluation video
    EVAL_RESET_OPTIONS = TRAIN_RESET_OPTIONS # Use same randomization for eval video, or define different ones

    # --- Setup Output Directories ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    log_path = os.path.join(OUTPUT_FOLDER, LOGS_SUBDIR)
    video_path_base = os.path.join(OUTPUT_FOLDER, VIDEOS_SUBDIR)
    plot_path_base = os.path.join(OUTPUT_FOLDER, PLOTS_SUBDIR)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(video_path_base, exist_ok=True)
    os.makedirs(plot_path_base, exist_ok=True)

    model_filepath = os.path.join(OUTPUT_FOLDER, MODEL_FILENAME)
    steps_filepath = os.path.join(OUTPUT_FOLDER, STEPS_FILENAME)

    # --- Create Vectorized Training Environment ---
    print("Creating vectorized training environment...")
    train_env_options = {
        "max_time": MAX_TIME,
        "obs_window": OBS_WINDOW,
        "control_logic_class": CONTROL_LOGIC,
        "reset_options": TRAIN_RESET_OPTIONS,
        "add_reward_wrapper": True, # Rewards needed for training
        "add_po_wrapper": True,     # PO observations needed for training
        "render_mode": None,        # No rendering during training
        # Add other create_quadruped_env args if needed
    }
    # Use SubprocVecEnv for parallel processing, DummyVecEnv for debugging
    vec_env = SubprocVecEnv([make_env(i, env_options=train_env_options) for i in range(NUM_ENVS)])

    print(f"Training Env Observation Space: {vec_env.observation_space}")
    print(f"Training Env Action Space: {vec_env.action_space}")

    # --- Initialize or Load Model ---
    start_step = 0
    if os.path.isfile(model_filepath):
        print(f"Loading existing model from: {model_filepath}")
        model = MODEL_CLASS.load(model_filepath, env=vec_env)
        if os.path.isfile(steps_filepath):
            try:
                with open(steps_filepath, 'r') as f:
                    start_step = int(f.read())
                print(f"Resuming training from step {start_step}")
            except ValueError:
                print("Warning: Could not read step count from steps.txt, starting from 0.")
                start_step = 0
        else:
            start_step = 0
    else:
        print("Initializing new model...")
        model = MODEL_CLASS(POLICY, vec_env, verbose=VERBOSE) # Add policy_kwargs if needed
        # Example for RecurrentPPO:
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, policy_kwargs=dict(lstm_hidden_size=64))

    # --- Setup Callbacks ---
    # Pass the output_folder for saving the continuous log
    reward_callback = RewardCallback(output_folder=log_path, real_time_flag=REAL_TIME_PLOT)

    # --- Training Loop ---
    print(f"Starting training loop from step {start_step}...")
    for i in range(start_step, start_step + NUM_LEARN_CALLS):
        print(f"\n--- Training Iteration {i+1}/{start_step + NUM_LEARN_CALLS} ---")

        # Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_LEARN,
                    callback=reward_callback,
                    reset_num_timesteps=False, # Continue timestep count across learn calls
                    **LEARN_KWARGS)

        # Save the model and current step
        print("Saving model...")
        model.save(model_filepath)
        with open(steps_filepath, 'w') as f:
            f.write(str(i + 1))

        # --- Process and Save Reward Data ---
        print("Processing and saving reward data...")
        if not reward_callback.data['rewards']:
             print("Warning: No reward data collected by callback.")
             continue # Skip plotting/saving if no data

        steps_collected = len(reward_callback.data['rewards'])
        training_steps_axis = range(steps_collected) # Use actual collected steps

        # Create DataFrame for plotting/saving
        df_list = [pd.DataFrame({'Training Steps': training_steps_axis})]
        if 'components' in reward_callback.data and reward_callback.data['components']:
             components_df = pd.DataFrame(reward_callback.data['components'])
             df_list.append(components_df)
        else:
             print("Warning: No reward components data collected.")

        reward_data = {
            'Reward': reward_callback.data['rewards'],
        }
        if 'std' in reward_callback.data and reward_callback.data['std']:
             reward_data['Std'] = reward_callback.data['std']
        else:
             # Assign default std if missing, maybe NaN or 0
             reward_data['Std'] = [np.nan] * steps_collected


        df_list.append(pd.DataFrame(reward_data))

        # Concatenate all parts
        try:
            data = pd.concat(df_list, axis=1)
            # Add condition column if needed by plot function
            data['Condition'] = 'Training'
            # Reorder columns if RewardCallback provides an order
            if hasattr(reward_callback, 'column_order') and reward_callback.column_order:
                 # Ensure all expected columns exist, add missing ones with NaN
                 for col in reward_callback.column_order:
                     if col not in data.columns:
                         data[col] = np.nan
                 data = data[reward_callback.column_order]

            # Save the dataframe
            log_csv_path = os.path.join(log_path, f'rewards_iter_{i}.csv')
            data.to_csv(log_csv_path, index=False)
            print(f"Reward data saved to {log_csv_path}")

            # Plot the rewards over training steps
            plot_save_path = os.path.join(plot_path_base, f'reward_plot_iter_{i}.png')
            plot_data_line([data], xaxis='Training Steps', value='Reward', condition='Condition',
                           smooth=max(1, steps_collected // 100), # Avoid smoothing=0
                           title=f'Training Reward (Iteration {i+1})',
                           output=plot_save_path)
            print(f"Reward plot saved to {plot_save_path}")
            plt.close() # Close the plot figure

            # Plot reward components if data exists
            if 'components' in reward_callback.data and reward_callback.data['components']:
                components_plot_path = os.path.join(plot_path_base, f'reward_components_plot_iter_{i}.html')
                plot_reward_components(data, output=components_plot_path)
                print(f"Reward components plot saved to {components_plot_path}")
            else:
                print("Skipping reward components plot (no data).")

        except Exception as e:
            print(f"Error processing/plotting reward data: {e}")
            import traceback
            traceback.print_exc()


        # --- Evaluate and Save Video ---
        print("Evaluating model and saving video...")
        eval_video_path = os.path.join(video_path_base, f'run_iter_{i}.mp4')

        # Create a single evaluation environment
        eval_env_options = {
            "max_time": EVAL_MAX_TIME,
            "obs_window": OBS_WINDOW,
            "control_logic_class": CONTROL_LOGIC,
            "reset_options": EVAL_RESET_OPTIONS,
            "add_reward_wrapper": True, # Include wrappers used during training
            "add_po_wrapper": True,
            "render_mode": "human",
            "save_video": True,
            "video_path": eval_video_path,
            # Ensure width/height match desired video output
            "width": 720,
            "height": 480,
        }
        eval_env = create_quadruped_env(**eval_env_options)

        try:
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            eval_step_count = 0
            while not done and not truncated:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                eval_env.render()
                done = terminated
                eval_step_count += 1

            print(f"Evaluation finished after {eval_step_count} steps. Video saved to {eval_video_path}")

        except Exception as e:
            print(f"Error during evaluation/video saving: {e}")
            import traceback
            traceback.print_exc()
        finally:
            eval_env.close() # Important to release video writer

    print("\n--- Training Complete ---")
    vec_env.close() # Close the vectorized training environments

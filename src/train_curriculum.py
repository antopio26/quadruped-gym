import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import RecurrentPPO # Uncomment if using LSTM
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import copy # Needed for deep copying options

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
    :param seed: the initial seed for RNG (handled by VecEnv)
    :param env_options: Dictionary of options for create_quadruped_env
    """
    if env_options is None:
        env_options = {}
    def _init():
        # Use a deep copy to prevent modifications in one process affecting others
        current_env_options = copy.deepcopy(env_options)
        # SB3 VecEnv handles seeding based on the initial seed + rank
        env = create_quadruped_env(**current_env_options)
        return env
    return _init

if __name__ == '__main__':
    # --- Configuration ---
    REAL_TIME_PLOT = False
    OUTPUT_FOLDER = './policies/po_vel_curriculum_sac_v3' # Choose a new folder name
    MODEL_FILENAME = 'policy.zip'
    STEPS_FILENAME = 'steps.txt' # Stores completed learn calls (iterations)
    LOGS_SUBDIR = 'logs'
    VIDEOS_SUBDIR = 'videos'
    PLOTS_SUBDIR = 'plots'

    # Environment parameters (Defaults, will be overridden by curriculum)
    OBS_WINDOW = 1
    MAX_TIME = 20.0
    NUM_ENVS = 10
    CONTROL_LOGIC = VelocityHeadingControls

    # Training parameters
    MODEL_CLASS = SAC
    POLICY = "MlpPolicy"
    TOTAL_TIMESTEPS_PER_LEARN = 500_000
    NUM_LEARN_CALLS = 100 # Total training iterations
    LEARN_KWARGS = {"progress_bar": True}
    VERBOSE = 0
   
    BASE_RANDOM_FORCE_OPTIONS = {
        'apply_translational_forces': True, 'translational_force_magnitude_range': (1.0, 10.0),
        'apply_rotational_forces': True, 'rotational_force_magnitude_range': (0.1, 0.3),
        'force_duration_range': (0.1, 0.5), 'force_interval_range': (1.0, 3.0),
        'force_body_name': "FRAME", 'apply_at_reset': False
    }

    # --- Curriculum Definition ---
    # Each stage defines parameters active *from* stage_start_iter onwards
    # until the next stage begins.
    curriculum_stages = [
        {
            "stage_start_iter": 0,
            "name": "Stage 1: Stand Still",
            "max_time": 5.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 15,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'fixed_speed': 0.0,
                    'fixed_velocity_angle': 0.0,
                    # 'fixed_heading_angle': 0.0
                    'max_theta': 15,
                }
            },
            "add_force_wrapper": False,
            "random_force_options": None
        },
        {
            "stage_start_iter": 5,
            "name": "Stage 2: Heading Control",
            "max_time": 8.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 180,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'fixed_speed': 0.0,
                    'fixed_velocity_angle': 0.0,
                    'max_theta': 180,
                }
            },
            "add_force_wrapper": False,
            "random_force_options": None
        },
        {
            "stage_start_iter": 20,
            "name": "Stage 3: Moderate Semi-Omni Walk",
            "max_time": 10.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 180,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'min_speed': 0.0,
                    'max_speed': 0.35,
                    'max_alpha': 45,
                    'max_theta': 180
                }
            },
            "add_force_wrapper": False,
            "random_force_options": None
        },
        {
            "stage_start_iter": 40,
            "name": "Stage 4: Semi-Omni Walk",
            "max_time": 15.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 180,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'min_speed': 0.0,
                    'max_speed': 0.5,
                    'max_alpha': 90,
                    'max_theta': 180
                }
            },
            "add_force_wrapper": False,
            "random_force_options": None
        },
        {
            "stage_start_iter": 60,
            "name": "Stage 5: Omni Walk",
            "max_time": 20.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 180,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'min_speed': 0.0,
                    'max_speed': 0.5,
                    'max_alpha': 180,
                    'max_theta': 180
                }
            },
            "add_force_wrapper": False,
            "random_force_options": None
        },
        {
            "stage_start_iter": 80,
            "name": "Stage 6: Omni Walk Forces",
            "max_time": 20.0,
            "reset_options": {
                'randomize_initial_state': True,
                'initial_state_options': {
                    'start_height': 0.25,
                    'max_z_axis_variation': 45,
                    'max_z_axis_rotation_angle': 180,
                    'max_linear_velocity': 0.1,
                    'max_angular_velocity': 0.1,
                    'randomize_joint_angles': True,
                    'friction_range': (0.5, 5.0)
                },
                'control_inputs_sampling_options': {
                    'min_speed': 0.0,
                    'max_speed': 0.5,
                    'max_alpha': 180,
                    'max_theta': 180
                }
            },
            "add_force_wrapper": True,
            "random_force_options": BASE_RANDOM_FORCE_OPTIONS
        },
    ]

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

    # --- Determine Start Step and Initial Curriculum Stage ---
    start_iter = 0 # Start from iteration 0
    if os.path.isfile(model_filepath) and os.path.isfile(steps_filepath):
        try:
            with open(steps_filepath, 'r') as f:
                start_iter = int(f.read())
            print(f"Resuming training from iteration {start_iter}")
        except ValueError:
            print("Warning: Could not read iteration count from steps.txt, starting from 0.")
            start_iter = 0
    else:
        start_iter = 0

    # Find the correct curriculum stage based on start_iter
    current_stage_index = 0
    for i, stage in enumerate(reversed(curriculum_stages)):
        if start_iter >= stage["stage_start_iter"]:
            current_stage_index = len(curriculum_stages) - 1 - i
            break
    print(f"Starting/Resuming at Curriculum Stage: {curriculum_stages[current_stage_index]['name']}")

    # --- Function to Create Vectorized Environment based on Stage ---
    def create_vec_env_for_stage(stage_index: int, num_envs: int):
        stage_params = curriculum_stages[stage_index]
        print(f"Creating VecEnv for Stage: {stage_params['name']}")
        env_opts = {
            "max_time": stage_params["max_time"],
            "obs_window": OBS_WINDOW,
            "control_logic_class": CONTROL_LOGIC,
            "reset_options": stage_params["reset_options"],
            "add_reward_wrapper": True,
            "add_po_wrapper": True,
            "add_force_wrapper": stage_params["add_force_wrapper"],
            "random_force_options": stage_params.get("random_force_options"), # Use .get for safety
            "render_mode": None,
        }
        # Use SubprocVecEnv for parallel processing, DummyVecEnv for debugging
        # Pass a *copy* of env_opts to make_env
        return SubprocVecEnv([make_env(i, env_options=env_opts) for i in range(num_envs)])
        # return DummyVecEnv([make_env(0, env_options=env_opts)]) # For debugging

    # --- Create Initial Vectorized Training Environment ---
    vec_env = create_vec_env_for_stage(current_stage_index, NUM_ENVS)

    print(f"Initial Training Env Observation Space: {vec_env.observation_space}")
    print(f"Initial Training Env Action Space: {vec_env.action_space}")

    # --- Initialize or Load Model ---
    if os.path.isfile(model_filepath):
        print(f"Loading existing model from: {model_filepath}")
        # Load the model, environment will be set later if needed or now
        model = MODEL_CLASS.load(model_filepath, env=vec_env)
        # If the environment changed significantly, you might want to reset the optimizer
        # model.load(model_filepath, env=vec_env, custom_objects={"learning_rate": new_lr, ...})
    else:
        print("Initializing new model...")
        model = MODEL_CLASS(POLICY, vec_env, verbose=VERBOSE)

    # --- Setup Callbacks ---
    reward_callback = RewardCallback(output_folder=log_path, real_time_flag=REAL_TIME_PLOT)

    # --- Training Loop ---
    print(f"Starting training loop from iteration {start_iter}...")
    total_iterations_to_run = NUM_LEARN_CALLS

    for i in range(start_iter, total_iterations_to_run):
        print(f"\n--- Training Iteration {i+1}/{total_iterations_to_run} ---")

        # --- Curriculum Stage Check ---
        next_stage_index = -1
        if current_stage_index + 1 < len(curriculum_stages):
            if i >= curriculum_stages[current_stage_index + 1]["stage_start_iter"]:
                next_stage_index = current_stage_index + 1

        if next_stage_index != -1:
            print(f"\nSwitching to Curriculum Stage {next_stage_index + 1}: {curriculum_stages[next_stage_index]['name']}")
            current_stage_index = next_stage_index

            # Close old env and create new one
            print("Closing old environment...")
            vec_env.close()
            print("Creating new environment for the current stage...")
            vec_env = create_vec_env_for_stage(current_stage_index, NUM_ENVS)

            # Update the model's environment
            print("Updating model's environment...")
            model.set_env(vec_env)
            # Optional: Reset optimizer if needed for drastic env changes
            # model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(), lr=model.learning_rate, **model.policy.optimizer_kwargs)
            print("Environment updated for model.")
        else:
            print(f"Continuing with Curriculum Stage: {curriculum_stages[current_stage_index]['name']}")

        # Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_LEARN,
                    callback=reward_callback,
                    reset_num_timesteps=False, # Keep counting timesteps across learn calls
                    **LEARN_KWARGS)

        # Save the model and current iteration number
        print("Saving model...")
        model.save(model_filepath)
        with open(steps_filepath, 'w') as f:
            f.write(str(i + 1)) # Save the number of the *completed* iteration

        # --- Process and Save Reward Data ---
        print("Processing and saving reward data...")
        # (Keep the existing reward processing and plotting logic here)
        if not reward_callback.data['rewards']:
             print("Warning: No reward data collected by callback.")
             continue

        steps_collected = len(reward_callback.data['rewards'])
        training_steps_axis = range(steps_collected)

        df_list = [pd.DataFrame({'Training Steps': training_steps_axis})]
        if 'components' in reward_callback.data and reward_callback.data['components']:
             components_df = pd.DataFrame(reward_callback.data['components'])
             df_list.append(components_df)
        else:
             print("Warning: No reward components data collected.")

        reward_data = {'Reward': reward_callback.data['rewards']}
        if 'std' in reward_callback.data and reward_callback.data['std']:
             reward_data['Std'] = reward_callback.data['std']
        else:
             reward_data['Std'] = [np.nan] * steps_collected

        df_list.append(pd.DataFrame(reward_data))

        try:
            data = pd.concat(df_list, axis=1)
            data['Condition'] = f'Training_Iter_{i}' # Add iteration info
            if hasattr(reward_callback, 'column_order') and reward_callback.column_order:
                 for col in reward_callback.column_order:
                     if col not in data.columns: data[col] = np.nan
                 # Ensure 'Condition' and 'Training Steps' are present if needed by plotting
                 if 'Condition' not in reward_callback.column_order: reward_callback.column_order.append('Condition')
                 if 'Training Steps' not in reward_callback.column_order: reward_callback.column_order.insert(0, 'Training Steps')
                 data = data[reward_callback.column_order]

            log_csv_path = os.path.join(log_path, f'rewards_iter_{i}.csv')
            data.to_csv(log_csv_path, index=False)
            print(f"Reward data saved to {log_csv_path}")

            # Plotting (consider plotting overall progress across iterations later)
            plot_save_path = os.path.join(plot_path_base, f'reward_plot_iter_{i}.png')
            plot_data_line([data], xaxis='Training Steps', value='Reward', condition='Condition',
                           smooth=max(1, steps_collected // 100),
                           title=f'Training Reward (Iteration {i+1})',
                           output=plot_save_path)
            print(f"Reward plot saved to {plot_save_path}")
            plt.close()

            if 'components' in reward_callback.data and reward_callback.data['components']:
                components_plot_path = os.path.join(plot_path_base, f'reward_components_plot_iter_{i}.html')
                # Ensure 'Training Steps' exists for plot_reward_components if it uses it
                if 'Training Steps' not in data.columns:
                    data['Training Steps'] = training_steps_axis
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

        # Get parameters from the current training stage for evaluation
        current_stage_params = curriculum_stages[current_stage_index]

        # Create a single evaluation environment (using defined eval settings)
        eval_env_options = {
            "max_time": current_stage_params["max_time"],
            "obs_window": OBS_WINDOW,
            "control_logic_class": CONTROL_LOGIC,
            "reset_options": current_stage_params["reset_options"],
            "add_reward_wrapper": True, # Keep wrappers consistent if needed for obs/info
            "add_po_wrapper": True,
            "add_force_wrapper": current_stage_params["add_force_wrapper"],
            "random_force_options": current_stage_params.get("random_force_options"),
            "render_mode": "human",     # Change to human for real-time display
            "save_video": True,        # Disable video saving when rendering human
            "video_path": eval_video_path,
            "width": 720, "height": 480,
        }

        # Create the environment directly for human rendering
        print("Creating direct evaluation environment with human rendering...")
        eval_env = create_quadruped_env(**eval_env_options)

        try:
            # Unpack the tuple returned by gym.Env.reset()
            obs, info = eval_env.reset()
            # For MuJoCo environments, render() might not be needed explicitly in the loop
            # if render_mode='human' is set during creation, but calling it ensures rendering.
            # If it renders automatically, you can remove the eval_env.render() call below.
            eval_env.render() # Initial render

            done = False
            eval_step_count = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True) # Use deterministic for eval
                obs, reward, terminated, truncated, info = eval_env.step(action)
                # Render is handled internally by the env when save_video=True and render_mode='rgb_array'
                # For render_mode='human', call eval_env.render() explicitly if needed.
                eval_env.render()
                done = terminated or truncated # Direct env returns booleans
                eval_step_count += 1

            print(f"Evaluation finished after {eval_step_count} steps.")
        except Exception as e:
            print(f"Error during evaluation/video saving: {e}")
            import traceback
            traceback.print_exc()
        finally:
            eval_env.close() # Important to release video writer and close env

    print("\n--- Training Complete ---")
    vec_env.close() # Close the final training environment
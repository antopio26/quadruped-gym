import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from stable_baselines3 import PPO

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

# Import the environment builder and control logic
from src.envs.env_builder import create_quadruped_env
from src.controls.velocity_heading_controls import VelocityHeadingControls

def evaluate_model(model_path: str, obs_window: int = 5):
    """
    Evaluates a trained SB3 model using the new wrapped environment structure.

    Args:
        model_path (str): Path to the trained SB3 model (.zip file).
        obs_window (int): Observation window size used during training.
                          Must match the model's expected input shape.
    """
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    print(f"Evaluating model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # --- Create the Environment using the Builder ---
    # We need PO wrapper if the model was trained with it.
    # We don't strictly need the reward wrapper for evaluation, but it doesn't hurt.
    # Set render_mode to "human".
    env = create_quadruped_env(
        render_mode="human",
        obs_window=obs_window, # Crucial: Match training
        control_logic_class=VelocityHeadingControls,
        add_reward_wrapper=True, # Include if rewards/termination are desired during eval
        add_po_wrapper=True      # Must match model training
        # Add other relevant env args if needed (e.g., max_time, model_path)
    )
    print(f"Environment Observation Space: {env.observation_space}")
    print(f"Environment Action Space: {env.action_space}")

    # --- Set Desired Control Inputs (Example) ---
    # Access the control logic via the wrapper's property
    # Use methods specific to VelocityHeadingControls
    try:
        # Example: Set a forward velocity and zero heading change
        target_speed = 0.2
        target_local_angle = 0.0 # Angle relative to heading (0 = straight)
        target_heading_angle = 0.0 # Global heading angle (0 = positive X axis)

        # Use the specific methods of the control logic instance
        env.unwrapped.get_wrapper_by_name("ControlInputWrapper").current_controls.set_orientation(target_heading_angle)
        env.unwrapped.get_wrapper_by_name("ControlInputWrapper").current_controls.set_velocity_speed_alpha(target_speed, target_local_angle)

        print(f"Set control inputs: Speed={target_speed}, Local Angle={target_local_angle}, Heading={target_heading_angle}")
        print(f"Resulting Global Velocity Command: {env.unwrapped.get_wrapper_by_name('ControlInputWrapper').current_controls.global_velocity[:2]}")

    except AttributeError as e:
        print(f"Error setting control inputs. Does the env have ControlInputWrapper with VelocityHeadingControls? {e}")
        env.close()
        return
    except Exception as e:
        print(f"An unexpected error occurred setting controls: {e}")
        env.close()
        return


    # --- Load the Model ---
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Check if the model architecture and observation space match the environment.")
        env.close()
        return

    # --- Run Evaluation Loop ---
    num_episodes = 1 # Evaluate for one episode
    all_rewards = []

    try:
        for i in range(num_episodes):
            print(f"\n--- Starting Evaluation Episode {i+1} ---")
            obs, info = env.reset()
            done = False
            truncated = False # Gymnasium uses terminated and truncated
            episode_rewards = []
            step_count = 0

            while not done and not truncated:
                # Check if viewer closed
                if env.unwrapped.render_mode == "human" and (env.unwrapped.viewer is None or not env.unwrapped.viewer.is_running()):
                     print("Viewer closed, stopping evaluation.")
                     done = True # Treat as end of episode
                     break

                action, _state = model.predict(obs, deterministic=True) # Use deterministic for eval
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated # Use terminated flag

                episode_rewards.append(reward)
                step_count += 1

                # Render is called internally by env.step() if render_mode is set

            print(f"Episode finished after {step_count} steps.")
            print(f"Total Reward: {sum(episode_rewards):.3f}")
            all_rewards.append(episode_rewards)

            # Plot the rewards for this episode
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(episode_rewards)), episode_rewards)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title(f'Evaluation Rewards Over Steps (Episode {i+1})')
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment.")
        env.close()

if __name__ == '__main__':
    # --- IMPORTANT: Set the correct path and obs_window ---
    MODEL_PATH = '../policies/po_new_ppo_v1/policy.zip' # <--- CHANGE THIS PATH
    OBS_WINDOW = 5 # <--- CHANGE THIS TO MATCH TRAINING

    # Construct absolute path
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, MODEL_PATH))

    evaluate_model(model_path=absolute_model_path, obs_window=OBS_WINDOW)

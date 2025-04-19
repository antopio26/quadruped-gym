import numpy as np
import time
import os
import sys
from stable_baselines3 import PPO # Or SAC, TD3, etc. depending on your model

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

# Import the environment builder and control logic
from src.envs.env_builder import create_quadruped_env
from src.controls.velocity_heading_controls import VelocityHeadingControls # Or the one used for training

def run_model_test(model_path: str, obs_window: int):
    """
    Interactively evaluates a trained SB3 model in the wrapped QuadrupedEnv.
    Runs indefinitely, prompting the user to reset or quit when an episode ends.
    Requires running with 'mjpython' on macOS for the interactive viewer.

    Args:
        model_path (str): Path to the trained SB3 model (.zip file).
        obs_window (int): Observation window size used during training.
    """
    print("Initializing environment and loading model for interactive evaluation...")
    print("NOTE: Run this script using 'mjpython test_model_interactive.py' on macOS.")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    env = None # Initialize env to None for finally block
    try:
        # --- Create the Environment using the Builder ---
        # Must match the configuration the model was trained with, especially obs_window.
        # Set render_mode="human". max_time=inf for continuous running.
        env = create_quadruped_env(
            render_mode="human",
            max_time=np.inf, # Run until terminated/truncated or viewer closed
            obs_window=obs_window, # CRUCIAL: Match training
            control_logic_class=VelocityHeadingControls, # Match training
            add_reward_wrapper=True, # Include wrappers model expects
            add_po_wrapper=True,
            # Add any other necessary env init args that match training
        )
        print(f"Environment Observation Space: {env.observation_space}")

        # --- Load the trained model ---
        # Ensure the model class (PPO, SAC, etc.) matches the saved model
        print(f"Loading model from: {model_path}")
        # Pass the created env to load (SB3 checks compatibility)
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")

        print("\nStarting evaluation loop...")
        print("Close the MuJoCo viewer window OR type 'q' at the prompt to stop.")

        # --- Initial Reset ---
        # Use a seed for reproducibility if desired, otherwise random
        obs, info = env.reset()#seed=42)

        # --- Main Interactive Loop ---
        while True:
            # --- Check Viewer Status ---
            # Access viewer status via unwrapped BaseQuadrupedEnv
            base_env = env.unwrapped # Get the innermost QuadrupedEnv instance
            if base_env.viewer is None:
                 print("Viewer not initialized or closed unexpectedly.")
                 break
            if not base_env.viewer.is_running():
                print("Viewer closed by user. Exiting.")
                break

            # --- Get Action from Model ---
            action, _states = model.predict(obs, deterministic=True) # Use deterministic=True for eval

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Render the Environment ---
            env.render()

            # --- Handle Episode End ---
            if terminated or truncated:
                print("-" * 30)
                print(f"Episode finished (Terminated: {terminated}, Truncated: {truncated})")
                # You can print info dict contents here if useful, e.g., info.get('reward_components')
                print("-" * 30)

                while True: # Loop to handle user input for reset/quit
                    # Check viewer again in case it was closed while waiting for input
                    if not base_env.viewer.is_running():
                        print("Viewer closed while waiting for input. Exiting.")
                        return # Exit the function cleanly

                    reset_input = input("Press Enter to reset episode, or type 'q' and Enter to quit: ").strip().lower()

                    if reset_input == 'q':
                        print("Quit command received. Exiting.")
                        return # Exit the function cleanly
                    elif reset_input == '':
                        print("Resetting environment...")
                        # Reset the environment, potentially with a new seed or options
                        obs, info = env.reset()#seed=np.random.randint(0, 10000))
                        print("Environment reset.")
                        break # Exit the input loop and continue the main simulation loop
                    else:
                        print("Invalid input. Please press Enter or type 'q'.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure env.close() is called reliably
        if env is not None:
             print("Closing environment...")
             env.close()
        print("Script finished.")

if __name__ == "__main__":
    # --- IMPORTANT: Set the correct path and obs_window for your trained model ---
    MODEL_TO_TEST = '../policies/po_new_ppo_v1/policy.zip' # <--- CHANGE THIS PATH
    OBS_WINDOW_USED_IN_TRAINING = 1 # <--- CHANGE THIS TO MATCH TRAINING

    # Construct absolute path relative to the script location
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, MODEL_TO_TEST))

    run_model_test(absolute_model_path, OBS_WINDOW_USED_IN_TRAINING)


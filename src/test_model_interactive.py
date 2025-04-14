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

from src.envs.po_quad import POWalkingQuadrupedEnv

def run_model_test(model_path):
    """
    Interactively evaluates a trained SB3 model in the POWalkingQuadrupedEnv.
    Runs indefinitely, prompting the user to reset or quit when an episode ends.
    Requires running with 'mjpython' on macOS.

    Args:
        model_path (str): Path to the trained SB3 model (.zip file).
    """
    print("Initializing environment and loading model for interactive evaluation...")
    print("NOTE: Run this script using 'mjpython test_model_interactive.py' on macOS.")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        # Initialize the environment
        # Set max_time to infinity for continuous running until termination/truncation
        env = POWalkingQuadrupedEnv(
            render_mode="human",
            max_time=np.inf, # No time limit per se, relies on term/trunc
            # Add any other necessary env init args here (must match training)
            obs_window=1 # Example: Ensure obs_window matches model training
        )

        # Load the trained model
        # Ensure the model class (PPO, SAC, etc.) matches the saved model
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")

        print("Starting evaluation loop...")
        print("Close the MuJoCo viewer window OR type 'q' at the prompt to stop.")

        # Reset the environment to get the initial state
        obs, info = env.reset()

        # Main loop
        while True:
            # Check if the viewer is still running. If not, exit.
            if env.viewer is None:
                 print("Viewer not initialized or closed unexpectedly.")
                 break
            if not env.viewer.is_running():
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
                print("-" * 30)
                while True: # Loop to handle user input
                    # Check viewer again in case it was closed while waiting for input
                    if not env.viewer.is_running():
                        print("Viewer closed while waiting for input. Exiting.")
                        # Need to break out of the outer loop as well
                        env.close() # Ensure cleanup before forced exit
                        return # Exit the function

                    reset_input = input("Press Enter to reset episode, or type 'q' and Enter to quit: ").strip().lower()
                    if reset_input == 'q':
                        print("Quit command received. Exiting.")
                        # Need to break out of the outer loop
                        env.close() # Ensure cleanup before forced exit
                        return # Exit the function
                    elif reset_input == '':
                        print("Resetting environment...")
                        obs, info = env.reset()
                        break # Exit the input loop and continue the main simulation loop
                    else:
                        print("Invalid input. Please press Enter or type 'q'.")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure env.close() is called even if it was closed earlier
        if 'env' in locals() and env is not None and env.viewer is not None and env.viewer.is_running():
             print("Closing environment...")
             env.close()
        elif 'env' in locals() and env is not None:
             # If viewer closed but env object exists, ensure other cleanup happens
             env.close()
        print("Script finished.")

if __name__ == "__main__":
    # --- IMPORTANT: Set the correct path to your trained model ---
    MODEL_TO_TEST = '../policies/po_new_ppo_v0/policy.zip' # <--- CHANGE THIS PATH

    # Construct absolute path relative to the script location
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, MODEL_TO_TEST))

    run_model_test(absolute_model_path)


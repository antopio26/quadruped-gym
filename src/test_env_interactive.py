import numpy as np
import time
import os
import sys

# --- Add project root to sys.path ---
# This assumes the script is run from the 'src' directory or the project root.
# Adjust the path depth ('..') if necessary based on where you place the script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

from src.envs.po_quad import POWalkingQuadrupedEnv

def run_env_test():
    """
    Interactively tests the POWalkingQuadrupedEnv with random actions.
    Runs indefinitely until the MuJoCo viewer window is closed.
    Requires running with 'mjpython' on macOS.
    """
    print("Initializing environment for interactive testing...")
    print("NOTE: Run this script using 'mjpython test_env_interactive.py' on macOS.")

    try:
        # Initialize the environment
        # Set max_time to infinity for continuous running
        env = POWalkingQuadrupedEnv(
            render_mode="human",
            max_time=np.inf, # No time limit
            # Add any other necessary env init args here, e.g., obs_window
            obs_window=1 # Example: Set obs_window if needed by PO env
        )
        print("Environment initialized. Starting simulation loop...")
        print("Close the MuJoCo viewer window to stop.")

        # Reset the environment to get the initial state
        obs, info = env.reset()

        # Main loop - runs until the viewer is closed
        while True:
            # Check if the viewer is still running. If not, exit.
            # This check needs to happen before accessing env.viewer potentially
            if env.viewer is None:
                 print("Viewer not initialized or closed unexpectedly.")
                 break
            if not env.viewer.is_running():
                print("Viewer closed by user. Exiting.")
                break

            # --- Take Action ---
            # Use random actions for testing the environment dynamics
            # action = env.action_space.sample()

            # Or use a specific action for testing
            # action = np.zeros(env.action_space.shape)

            # Or set no action
            action = None

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Render the Environment ---
            # The render method in base_quad handles viewer sync and timing
            env.render()

            # --- Handle Episode End (Optional but good practice) ---
            # Even with max_time=inf, other termination conditions might exist
            if terminated or truncated:
                print(f"Episode ended (Terminated: {terminated}, Truncated: {truncated}). Resetting environment.")
                obs, info = env.reset()

            # Optional small delay if needed, though viewer sync should manage speed
            # time.sleep(0.01)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if 'env' in locals() and env is not None:
            print("Closing environment...")
            env.close()
        print("Script finished.")

if __name__ == "__main__":
    run_env_test()

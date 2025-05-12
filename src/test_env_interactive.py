import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

# Import the base environment class directly
from src.envs.base_quad import QuadrupedEnv

from src.envs.env_builder import create_quadruped_env
from src.controls.velocity_heading_controls import VelocityHeadingControls

def run_env_test():
    """
    Interactively tests the base QuadrupedEnv with zero actions.
    Runs indefinitely until the MuJoCo viewer window is closed.
    Requires running with 'mjpython' on macOS.
    """
    print("Initializing base environment for interactive testing...")
    print("NOTE: Run this script using 'mjpython test_env_interactive.py' on macOS.")

    env = None # Initialize env to None for finally block
    try:
        # --- Initialize the Base Environment ---
        # Set render_mode="human" and max_time=inf for continuous running.
        OBS_WINDOW = 1 # Observation window size (1 for single timestep, >1 for history)
        MAX_TIME = np.inf # Max seconds per episode
        CONTROL_LOGIC = VelocityHeadingControls

        TEST_RESET_OPTIONS = {
            'randomize_initial_state': True, # Randomize base pose/velocity in BaseQuadrupedEnv
            'initial_state_options': {
                'friction_range': (3, 3), # Friction range for the ground
            },
            'control_inputs': {             # Options for ControlInputWrapper.sample
                'min_speed': 0.0,
                'max_speed': 0.5,
                'fixed_heading_angle': None,
                'fixed_velocity_angle': None,
                'fixed_speed': None
            }
        }
        RANDOM_FORCE_OPTIONS = {
            'apply_translational_forces': True,
            'translational_force_magnitude_range': (5.0, 20.0),
            'apply_rotational_forces': True,
            'rotational_force_magnitude_range': (0.5, 2.0),
            'force_duration_range': (0.1, 0.5), # Shorter duration for force vs impulse
            'force_interval_range': (1.0, 5.0),
            'force_body_name': "FRAME",
            'apply_at_reset': False
        }
        test_env_options = {
            "max_time": MAX_TIME,
            "obs_window": OBS_WINDOW,
            "control_logic_class": CONTROL_LOGIC,
            "reset_options": TEST_RESET_OPTIONS,
            "add_reward_wrapper": True, # Include wrappers used during training
            "add_po_wrapper": False,
            "add_force_wrapper": False,
            "random_force_options": RANDOM_FORCE_OPTIONS,
            "render_mode": "human",
        }
        env = create_quadruped_env(**test_env_options)
        
        # QuadrupedEnv(
        #     render_mode="human",
        #     max_time=np.inf, # No time limit
            # Add any other necessary base env init args here
            # e.g., model_path if not default
        #    reset_options={'randomize_initial_state': True} # Start from random poses
        #)

        # --- Alternative: Test with Wrappers ---
        # Uncomment below to test the fully wrapped environment interactively
        # print("Initializing wrapped environment for interactive testing...")
        # env = create_quadruped_env(
        #     render_mode="human",
        #     max_time=np.inf,
        #     obs_window=1, # Or any value, obs not used here
        #     control_logic_class=VelocityHeadingControls,
        #     add_reward_wrapper=True, # Include wrappers for full test
        #     add_po_wrapper=True,
        #     reset_options={'randomize_initial_state': True, 'control_inputs': {'max_speed': 0.2}}
        # )
        # -----------------------------------------

        print("Environment initialized. Starting simulation loop...")
        print("Close the MuJoCo viewer window to stop.")

        # --- Initial Reset ---
        obs, info = env.reset() # Get initial state

        # --- Add counter for steps ---
        step_count = 0

        # --- Main Interactive Loop ---
        while True:
            # --- Check Viewer Status ---
            # Access viewer status via unwrapped BaseQuadrupedEnv
            # If using create_quadruped_env, env.unwrapped gets the base env.
            # If using QuadrupedEnv directly, env is the base env.
            base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            if base_env.viewer is not None and not base_env.viewer.is_running():
                print("Viewer closed by user. Exiting.")
                break

            # --- Take Action ---
            # Use zero actions to observe passive dynamics or stability
            # action = np.zeros(env.action_space.shape)

            # Or use random actions to see how it reacts
            # action = env.action_space.sample()

            # Or pass none action to disable control
            # action = None

            # if step_count < 50:
            #     action = None # No action for the first 10 steps
            # else:
            #     action = np.array([1] * env.action_space.shape[0], dtype=np.float32) # Zero action

            action = np.array([0.0, 0.0, -0.5] * 4, dtype=np.float32) # Zero action

            # --- Step the Environment ---
            # We don't need the returned values for this simple test
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Render the Environment ---
            env.render()

            # --- Handle Episode End (Optional but good practice) ---
            # Even with max_time=inf, termination conditions might exist (e.g., falling in WalkingRewardWrapper)
            if terminated or truncated:
                print("-" * 30)
                print(f"Episode ended (Terminated: {terminated}, Truncated: {truncated}).")
                # Optionally print info dict contents
                print("Resetting environment...")
                obs, info = env.reset()
                step_count = 0
                print("-" * 30)

                break

            # --- Increment Step Count ---
            step_count += 1

        # --- End of Loop ---
        print("Simulation loop ended.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            print("Closing environment...")
            env.close()
        print("Script finished.")

if __name__ == "__main__":
    run_env_test()

import numpy as np
import time
import os
import sys
import hid # Use hid library

from stable_baselines3 import PPO, SAC, TD3 # etc. depending on your model

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path addition ---

# Import the environment builder and control logic
from src.envs.env_builder import create_quadruped_env
from src.controls.velocity_heading_controls import VelocityHeadingControls # Or the one used for training
from src.utils.envs import find_wrapper_by_name

# --- Gamepad Control Configuration ---
# Adjust these values based on your gamepad and preferences
DEADZONE = 0.1 # Deadzone for normalized [-1, 1] values
MAX_SPEED = 0.6  # Max linear speed component (m/s) for both X and Y local velocity
MAX_ANGULAR_VELOCITY = np.pi / 1.5 # Max heading angular velocity (radians/s)

# --- HID Device Configuration (e.g., PS4 Controller) ---
VENDOR_ID  = 0x054c # Replace with your gamepad's Vendor ID
PRODUCT_ID = 0x05c4 # Replace with your gamepad's Product ID

# --- Axis Mapping Configuration ---
# Define constants for the *purpose* of each axis
PURPOSE_VX = 1 # Controls local vx (Forward/Backward velocity)
PURPOSE_VY = 2 # Controls local vy (Strafe velocity)
PURPOSE_HEADING_RATE = 3 # Controls heading rate (Turning rate)
PURPOSE_UNUSED = 0

# Map HID report indices (1-based for rpt[1], rpt[2]...) to their purpose
# Default PS4 mapping:
AXIS_PURPOSE_MAPPING = {
    1: PURPOSE_HEADING_RATE, # rpt[1] (Left Stick X) controls Turning Rate
    2: PURPOSE_UNUSED,       # rpt[2] (Left Stick Y) is unused
    3: PURPOSE_VY,           # rpt[3] (Right Stick X) controls Strafe Velocity
    4: PURPOSE_VX,           # rpt[4] (Right Stick Y) controls Forward/Back Velocity
}


# Configure axis inversion based on the *purpose*
AXIS_INVERSION = {
    PURPOSE_VX: True,
    PURPOSE_VY: True,
    PURPOSE_HEADING_RATE: True,
    PURPOSE_UNUSED: False, # Doesn't matter, but include for completeness
}
# --- End of Gamepad Config ---

def normalize_hid_axis(raw_value):
    """Normalizes a raw HID axis value (0-255) to [-1, 1]."""
    return (raw_value - 128.0) / 128.0

def wrap_angle(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_control_description(mapping):
    """Generates a string describing the current control mapping."""
    vx_desc = "Not Mapped"
    vy_desc = "Not Mapped"
    hr_desc = "Not Mapped"

    for idx, purpose in mapping.items():
        stick_name = "Unknown Stick"
        axis_name = "Unknown Axis"
        if idx == 1: stick_name, axis_name = "Left Stick", "X"
        elif idx == 2: stick_name, axis_name = "Left Stick", "Y"
        elif idx == 3: stick_name, axis_name = "Right Stick", "X"
        elif idx == 4: stick_name, axis_name = "Right Stick", "Y"
        # Add more indices if needed

        full_name = f"{stick_name} {axis_name} (rpt[{idx}])"

        if purpose == PURPOSE_VX:
            vx_desc = full_name
        elif purpose == PURPOSE_VY:
            vy_desc = full_name
        elif purpose == PURPOSE_HEADING_RATE:
            hr_desc = full_name

    return f"Forward/Back: {vx_desc}, Strafe: {vy_desc}, Turn: {hr_desc}"


def run_model_test(model_path: str, obs_window: int):
    """
    Interactively evaluates a trained SB3 model using gamepad controls
    read via the 'hid' library, with configurable axis mapping using purpose constants.
    """
    print("Initializing environment, model, and HID gamepad...")
    print("NOTE: Run this script using 'mjpython test_model_interactive.py' on macOS.")
    print(f"Attempting to connect to HID device (Vendor: {VENDOR_ID:#06x}, Product: {PRODUCT_ID:#06x})...")

    gamepad = None
    try:
        gamepad = hid.device()
        gamepad.open(VENDOR_ID, PRODUCT_ID)
        gamepad.set_nonblocking(True)
        print(f"Successfully connected to HID device: {gamepad.get_product_string()}")
    except Exception as e:
        print(f"Error: Could not open HID device (Vendor: {VENDOR_ID:#06x}, Product: {PRODUCT_ID:#06x}).")
        print(f"Details: {e}")
        print("Ensure the gamepad is connected and the VENDOR_ID/PRODUCT_ID are correct.")
        if gamepad:
            gamepad.close()
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        if gamepad:
            gamepad.close()
        return

    env = None
    current_heading_angle = 0.0
    # Initialize normalized control values based on purpose
    norm_vx, norm_vy, norm_heading_rate = 0.0, 0.0, 0.0

    try:
        # --- Initialize the Base Environment ---
        OBS_WINDOW = obs_window
        MAX_TIME = np.inf
        CONTROL_LOGIC = VelocityHeadingControls

        TEST_RESET_OPTIONS = {
            'randomize_initial_state': False,
            'initial_state_options': { 'friction_range': (3, 3) },
            'control_inputs_sampling_options': {
                 'max_speed': 0.0, 'fixed_heading_angle': 0.0,
                 'fixed_velocity_angle': 0.0, 'fixed_speed': 0.0
            }
        }
        RANDOM_FORCE_OPTIONS = {
            'apply_translational_forces': False, 'apply_rotational_forces': False,
            'translational_force_magnitude_range': (5.0, 20.0),
            'rotational_force_magnitude_range': (0.5, 2.0),
            'force_duration_range': (0.1, 0.5), 'force_interval_range': (1.0, 5.0),
            'force_body_name': "FRAME", 'apply_at_reset': False
        }
        test_env_options = {
            "max_time": MAX_TIME, "obs_window": OBS_WINDOW,
            "control_logic_class": CONTROL_LOGIC, "reset_options": TEST_RESET_OPTIONS,
            "add_reward_wrapper": True, "add_po_wrapper": True,
            "add_force_wrapper": False, "random_force_options": RANDOM_FORCE_OPTIONS,
            "render_mode": "human",
        }

        env = create_quadruped_env(**test_env_options)
        dt = env.unwrapped.get_dt()
        print(f"Environment Observation Space: {env.observation_space}")
        print(f"Simulation timestep (dt): {dt}")

        # --- Load the trained model ---
        print(f"Loading model from: {model_path}")
        model = SAC.load(model_path, env=env)
        print("Model loaded successfully.")

        # --- Get Control Logic Instance ---
        try:
            control_input_wrapper = find_wrapper_by_name(env, "ControlInputWrapper")
            if control_input_wrapper is None: raise ValueError("ControlInputWrapper not found.")
            controls = control_input_wrapper.current_controls
            if not isinstance(controls, VelocityHeadingControls): raise TypeError("Incorrect control logic.")
            print("Successfully accessed VelocityHeadingControls.")
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Error accessing control logic: {e}")
            return

        print("\nStarting evaluation loop...")
        print(f"Control the robot using '{gamepad.get_product_string()}'.")
        print(f"Gamepad Control Mapping: {get_control_description(AXIS_PURPOSE_MAPPING)}")
        print("Close the MuJoCo viewer window OR type 'q' at the prompt to stop.")

        # --- Initial Reset ---
        obs, info = env.reset()
        current_heading_angle = 0.0
        controls.set_orientation(current_heading_angle)
        controls.set_velocity_xy(0.0, 0.0) # Use set_velocity_xy

        # --- Main Interactive Loop ---
        while True:
            start_time = time.time()

            # --- Check Viewer Status ---
            base_env = env.unwrapped
            if base_env.viewer is None:
                 print("Viewer not initialized or closed unexpectedly.")
                 break
            if not base_env.viewer.is_running():
                print("Viewer closed by user. Exiting.")
                break

            # --- Get Gamepad Input (using hid) ---
            rpt = gamepad.read(64)
            if rpt:
                # Reset normalized values before processing new report
                norm_vx, norm_vy, norm_heading_rate = 0.0, 0.0, 0.0
                # Process axes based on mapping
                for axis_index, purpose in AXIS_PURPOSE_MAPPING.items():
                    if purpose == PURPOSE_UNUSED: continue # Skip unused axes
                    if axis_index < len(rpt): # Check if index is valid for the report
                        raw_value = rpt[axis_index]
                        normalized_value = normalize_hid_axis(raw_value)

                        # Assign to the correct normalized variable based on purpose
                        if purpose == PURPOSE_VX:
                            norm_vx = normalized_value
                        elif purpose == PURPOSE_VY:
                            norm_vy = normalized_value
                        elif purpose == PURPOSE_HEADING_RATE:
                            norm_heading_rate = normalized_value
            # If no report, normalized values retain their previous state

            # --- Apply Deadzone ---
            current_vx = 0.0 if abs(norm_vx) < DEADZONE else norm_vx
            current_vy = 0.0 if abs(norm_vy) < DEADZONE else norm_vy
            current_heading_rate = 0.0 if abs(norm_heading_rate) < DEADZONE else norm_heading_rate

            # --- Apply Inversion based on purpose ---
            if AXIS_INVERSION.get(PURPOSE_VX, False): current_vx *= -1
            if AXIS_INVERSION.get(PURPOSE_VY, False): current_vy *= -1
            if AXIS_INVERSION.get(PURPOSE_HEADING_RATE, False): current_heading_rate *= -1

            # --- Map to Control Values ---
            target_vx = current_vx * MAX_SPEED
            target_vy = current_vy * MAX_SPEED
            heading_angular_velocity = current_heading_rate * MAX_ANGULAR_VELOCITY

            # --- Update Heading Angle ---
            current_heading_angle += heading_angular_velocity * dt
            current_heading_angle = wrap_angle(current_heading_angle)

            # --- Update Control Logic ---
            controls.set_orientation(current_heading_angle)
            controls.set_velocity_xy(target_vx, target_vy) # Use set_velocity_xy

            # --- Get Action from Model ---
            action, _states = model.predict(obs, deterministic=False)

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Render the Environment ---
            env.render()

            # --- Frame Rate Limiting ---
            elapsed_time = time.time() - start_time
            sleep_time = max(0, dt - elapsed_time)
            time.sleep(sleep_time) # Use the calculated sleep time

            # --- Handle Episode End ---
            if terminated or truncated:
                print("\n" + "-" * 30)
                print(f"Episode finished (Terminated: {terminated}, Truncated: {truncated})")
                print("-" * 30)
                while True:
                    if not base_env.viewer.is_running():
                        print("Viewer closed while waiting for input. Exiting.")
                        return # Exit function if viewer closed
                    reset_input = input("Press Enter to reset episode, or type 'q' and Enter to quit: ").strip().lower()
                    if reset_input == 'q':
                        print("Quit command received. Exiting.")
                        return # Exit function on quit command
                    elif reset_input == '':
                        print("Resetting environment...")
                        obs, info = env.reset()
                        current_heading_angle = 0.0
                        controls.set_orientation(current_heading_angle)
                        controls.set_velocity_xy(0.0, 0.0) # Use set_velocity_xy
                        # Reset normalized control values on reset
                        norm_vx, norm_vy, norm_heading_rate = 0.0, 0.0, 0.0
                        print("Environment reset.")
                        break # Break inner loop to continue outer simulation loop
                    else:
                        print("Invalid input. Please press Enter or type 'q'.")

    except Exception as e:
        print(f"\nAn error occurred during the simulation loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
             print("Closing environment...")
             env.close()
        if gamepad is not None:
            print("Closing HID device...")
            gamepad.close()
        print("Script finished.")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update VENDOR_ID and PRODUCT_ID above for your specific gamepad
    # You can find these using tools like 'lsusb' on Linux or System Information on macOS
    # Configure AXIS_PURPOSE_MAPPING and AXIS_INVERSION above as needed.

    # Path to the trained model policy
    MODEL_TO_TEST = '../policies/exam_test_v0/policy.zip' # UPDATE THIS PATH

    # Observation window size used during the training of the loaded model
    OBS_WINDOW_USED_IN_TRAINING = 1 # UPDATE THIS VALUE if different

    # --- Run the test ---
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, MODEL_TO_TEST))
    run_model_test(absolute_model_path, OBS_WINDOW_USED_IN_TRAINING)
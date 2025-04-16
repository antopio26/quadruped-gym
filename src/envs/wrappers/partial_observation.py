# src/envs/wrappers/partial_observation.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Tuple

from src.envs.wrappers.control_input import ControlInputWrapper
from src.envs.wrappers.utils import find_wrapper_by_name

# AHRS for orientation estimation
try:
    from ahrs.filters import Madgwick
    from ahrs.common import Quaternion
    _AHRS_AVAILABLE = True
except ImportError:
    _AHRS_AVAILABLE = False
    # Define dummy classes if ahrs is not installed
    class Madgwick:
        def __init__(self, *args, **kwargs): pass
        def updateIMU(self, *args, **kwargs): return np.array([1., 0., 0., 0.])
    class Quaternion:
        def __init__(self, *args, **kwargs): pass
        def to_angles(self): return np.zeros(3)
    print("Warning: 'ahrs' library not found. Orientation estimation in PartialObservationWrapper will be disabled (returning zeros).")


class PartialObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a quadruped environment to provide a partially observable state.

    Constructs observations using IMU data (accelerometer, gyroscope),
    estimated orientation (via Madgwick filter if 'ahrs' is installed),
    proprioceptive information (actuator commands, command derivatives),
    and external commands (from the info dict provided by ControlInputWrapper).
    Supports observation stacking using a rolling window.
    """

    class_name = "PartialObservationWrapper"

    def __init__(self,
                 env: gym.Env,
                 obs_window: int = 1,
                 expected_control_obs_size: Optional[int] = None):
        """
        Args:
            env: The environment to wrap (should provide base QuadrupedEnv methods).
            obs_window: Number of historical observation frames to stack.
            expected_control_obs_size: The expected size of 'control_inputs_obs'
                in the info dict. If None, it tries to infer from the wrapped env.
        """
        super().__init__(env)

        if obs_window < 1:
            raise ValueError("obs_window must be >= 1")
        self.obs_window = obs_window

        if not _AHRS_AVAILABLE and obs_window > 0:
            print("Warning: 'ahrs' library not installed. Orientation component of PO observation will be zero.")

        # --- Determine expected external control size ---
        if expected_control_obs_size is None:
            # Try to get from ControlInputWrapper if present using the utility function
            control_wrapper = find_wrapper_by_name(self, "ControlInputWrapper")

            if control_wrapper is not None and hasattr(control_wrapper, '_control_obs_size'):
                self._expected_control_obs_size = control_wrapper._control_obs_size
            else:
                # If not found or lacks attribute, default to 0
                print("Warning: Could not determine expected_control_obs_size from ControlInputWrapper. Assuming 0.")
                self._expected_control_obs_size = 0
        else:
                self._expected_control_obs_size = expected_control_obs_size

        # --- Define the Partial Observation Space ---
        single_obs_size = 0
        single_obs_size += 3  # Gyroscope
        single_obs_size += 3  # Accelerometer
        single_obs_size += 3  # Euler angles (estimated orientation)
        single_obs_size += 2  # Body velocity XY (proxy for optical flow)
        single_obs_size += self.env.action_space.shape[0] # Current control command
        single_obs_size += self.env.action_space.shape[0] # Control command derivative
        single_obs_size += self._expected_control_obs_size # External controls (from info)

        total_obs_size = single_obs_size * self.obs_window
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # --- State for PO ---
        self.madgwick_filter = Madgwick() # Dt and q0 set in reset
        self.computed_orientation = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._obs_buffer = deque(maxlen=self.obs_window)
        # Store previous control state *within this wrapper* for derivative calculation
        self.previous_ctrl_po = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        # Ensure reset is called to initialize buffer correctly
        # self.reset() # Calling reset in init is problematic; do it externally or on first use

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and PO-specific state including the observation buffer."""
        # Reset wrapped environment(s)
        observation, info = self.env.reset(seed=seed, options=options)
        # Note: 'observation' here is the full observation from the wrapped env, which we ignore.

        # --- Reset PO-Specific State ---
        dt = self.env.unwrapped.get_dt()
        q0 = self.env.unwrapped.get_body_orientation_quat().astype(np.float64)
        self.computed_orientation = q0
        if _AHRS_AVAILABLE:
            self.madgwick_filter = Madgwick(Dt=dt, q0=q0)
        else:
            self.madgwick_filter = Madgwick() # Dummy filter

        # Reset previous control based on state *after* reset
        self.previous_ctrl_po = self.env.unwrapped.get_control_inputs()

        # --- Initialize Observation Buffer ---
        self._obs_buffer.clear()
        # Compute the *actual* first partial observation frame using the reset state
        first_frame = self._compute_single_po_frame(info) # Pass info from reset

        # Fill the buffer by repeating the first frame
        for _ in range(self.obs_window):
            self._obs_buffer.append(first_frame)

        # Concatenate the filled buffer to get the initial stacked observation
        initial_stacked_observation = np.concatenate(list(self._obs_buffer), axis=0).astype(np.float32)

        # Verify shape
        if initial_stacked_observation.shape != self.observation_space.shape:
             raise ValueError(f"Initial PO observation shape mismatch! Expected {self.observation_space.shape}, got {initial_stacked_observation.shape}")

        return initial_stacked_observation, info # Return correct initial obs and info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps the environment and computes the new partial observation."""
        # Store control state *before* step for derivative calculation
        ctrl_before_step = self.env.unwrapped.get_control_inputs()

        # Step the wrapped environment(s)
        _observation, reward, terminated, truncated, info = self.env.step(action)
        # We ignore _observation as we compute our own PO observation below.

        # Compute the new PO observation frame using state *after* the step
        # Pass the info dict which should contain 'control_inputs_obs'
        po_observation_frame = self._compute_single_po_frame(info)

        # Update previous control state *after* using it for derivative
        self.previous_ctrl_po = ctrl_before_step # Use the control active during this step

        # Add the new frame to the buffer (deque handles the rolling window)
        self._obs_buffer.append(po_observation_frame)

        # Concatenate frames in the deque to form the final observation
        final_observation = np.concatenate(list(self._obs_buffer), axis=0).astype(np.float32)

        # Verify shape (optional, for debugging)
        # if final_observation.shape != self.observation_space.shape:
        #     print(f"Warning: Step PO observation shape mismatch! Expected {self.observation_space.shape}, got {final_observation.shape}")

        return final_observation, reward, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns the potentially stacked partial observation.

        Note: This method is called by the ObservationWrapper base class *after*
        step() returns. Since our PO calculation depends on state computed *during*
        step (like the info dict and Madgwick updates), we perform the full
        calculation within step() and return the final result there. This override
        is mainly to satisfy the base class structure but returns the already
        computed and buffered observation.
        """
        if not self._obs_buffer:
             # Should not happen if reset was called correctly
             print("Warning: Observation buffer empty in observation(). Returning zeros.")
             return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        # Return the latest stacked observation from the buffer
        return np.concatenate(list(self._obs_buffer), axis=0).astype(np.float32)


    def _compute_single_po_frame(self, info: Dict[str, Any]) -> np.ndarray:
        """Helper to compute a single frame of the partial observation."""
        # --- Get Sensor Data using Public Accessors ---
        # Access the *unwrapped* base environment for direct sensor methods
        base_env = self.env.unwrapped
        accel = base_env.get_body_linear_acceleration()
        gyro = base_env.get_body_angular_velocity()
        body_vel_xy = base_env.get_body_linear_velocity()[:2]
        current_ctrl = base_env.get_control_inputs() # Controls applied in the *last* mj_step

        # --- Estimate Orientation ---
        if _AHRS_AVAILABLE:
            # Update Madgwick filter state (mutates self.computed_orientation)
            self.computed_orientation = self.madgwick_filter.updateIMU(
                q=self.computed_orientation, gyr=gyro, acc=accel
            )
            euler_angles = Quaternion(self.computed_orientation).to_angles()
        else:
            euler_angles = np.zeros(3) # Placeholder if ahrs not available

        # --- Get External Control Observation from Info Dict ---
        external_control_obs = info.get('control_inputs_obs', None)
        if external_control_obs is None:
             print(f"Warning: 'control_inputs_obs' not found in info dict.")
             external_control_obs = np.array([]) # Empty array if not found
        elif len(external_control_obs) != self._expected_control_obs_size:
             print(f"Warning: Mismatch in 'control_inputs_obs' size. Expected {self._expected_control_obs_size}, got {len(external_control_obs)}. Padding/truncating.")
             padded_obs = np.zeros(self._expected_control_obs_size, dtype=np.float32)
             copy_len = min(len(external_control_obs), self._expected_control_obs_size)
             padded_obs[:copy_len] = external_control_obs[:copy_len]
             external_control_obs = padded_obs
        external_control_obs = external_control_obs.astype(np.float32)


        # --- Calculate Control Derivative ---
        # Uses self.previous_ctrl_po (controls from start of the step)
        dt = base_env.get_dt()
        if dt < 1e-9:
            ctrl_derivative = np.zeros_like(current_ctrl)
        else:
            # Derivative uses control state *before* step vs control state *before previous* step
            ctrl_derivative = (current_ctrl - self.previous_ctrl_po) / dt

        # --- Concatenate Observation Frame ---
        single_frame = np.concatenate([
            gyro,                   # 3
            accel,                  # 3
            euler_angles,           # 3
            body_vel_xy,            # 2
            current_ctrl,           # nu
            ctrl_derivative,        # nu
            external_control_obs    # expected_control_obs_size
        ]).astype(np.float32)

        # --- Verification (Optional) ---
        expected_single_size = self.observation_space.shape[0] // self.obs_window
        if single_frame.shape[0] != expected_single_size:
             # This indicates a mismatch in size calculation in __init__
             raise ValueError(f"Computed single PO frame size mismatch! Expected {expected_single_size}, got {single_frame.shape[0]}. Check __init__ calculation.")

        return single_frame


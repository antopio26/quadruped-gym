# src/wrappers/random_force_disturb.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Optional, Tuple, Dict, Any

class RandomForceDisturbWrapper(gym.Wrapper):
    """
    Applies random translational and/or rotational forces to a specified
    body in the MuJoCo environment for a random duration at random intervals.
    Uses `data.xfrc_applied`.

    Args:
        env: The Gymnasium environment to wrap (must have `model`, `data`, and `np_random` attributes).
        apply_translational_forces: If True, apply random forces (affect linear velocity).
        translational_force_magnitude_range: Tuple (min, max) magnitude for translational forces (N).
        apply_rotational_forces: If True, apply random torques (affect angular velocity).
        rotational_force_magnitude_range: Tuple (min, max) magnitude for rotational forces (N*m).
        force_duration_range: Tuple (min, max) seconds over which the sampled force is applied.
        force_interval_range: Tuple (min, max) seconds between the *start* of force events.
        force_body_name: Name of the body in the XML to apply forces to.
        apply_at_reset: If True, starts applying a force immediately upon reset.
    """
    def __init__(self,
                 env: gym.Env,
                 # Translational Force Params
                 apply_translational_forces: bool = True,
                 translational_force_magnitude_range: Tuple[float, float] = (5.0, 20.0),
                 # Rotational Force Params
                 apply_rotational_forces: bool = False,
                 rotational_force_magnitude_range: Tuple[float, float] = (0.5, 2.0),
                 # Timing Params
                 force_duration_range: Tuple[float, float] = (0.1, 0.5),
                 force_interval_range: Tuple[float, float] = (1.0, 5.0),
                 # Common Params
                 force_body_name: str = "FRAME",
                 apply_at_reset: bool = False):

        super().__init__(env)

        # Basic validation
        if not hasattr(env, 'model') or not hasattr(env, 'data'):
            raise ValueError("Wrapped environment must have 'model' and 'data' attributes (MuJoCo environment).")
        # Note: We rely on the base Env/Wrapper to provide self.np_random after reset

        # Store parameters
        self.apply_translational_forces = apply_translational_forces
        self.translational_force_magnitude_range = translational_force_magnitude_range
        self.apply_rotational_forces = apply_rotational_forces
        self.rotational_force_magnitude_range = rotational_force_magnitude_range
        self.force_duration_range = force_duration_range
        self.force_interval_range = force_interval_range
        self.force_body_name = force_body_name
        self.apply_at_reset = apply_at_reset

        if not apply_translational_forces and not apply_rotational_forces:
            print("Warning: Both translational and rotational forces are disabled in RandomForceDisturbWrapper.")

        # Get the ID of the target body
        self._force_body_id: Optional[int] = None
        try:
            # Access model through self.env after super().__init__()
            body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.force_body_name)
            if body_id == -1:
                print(f"Warning: Force body '{self.force_body_name}' not found in model. Random forces disabled in wrapper.")
            else:
                self._force_body_id = body_id
        except ValueError:
             print(f"Warning: Error finding force body '{self.force_body_name}'. Random forces disabled in wrapper.")
        except AttributeError:
             print("Warning: Could not access self.env.model during init. Body ID lookup failed.")


        # Internal state for force application
        self._next_force_event_time: float = float('inf') # Time to potentially start a new force
        self._force_end_time: float = float('-inf')      # Time the current force should stop
        self._current_force_vector = np.zeros(6)         # [fx, fy, fz, tx, ty, tz]

    # Removed the np_random property - rely on self.np_random from base class

    def _schedule_next_force_event(self, current_time: float):
        """Schedules the time for the next potential force application."""
        if self._force_body_id is None or (not self.apply_translational_forces and not self.apply_rotational_forces):
            self._next_force_event_time = float('inf')
            return
        if self.np_random is None: # Check if RNG is available (might not be before first reset)
             print("Warning: np_random not available in _schedule_next_force_event. Cannot schedule.")
             self._next_force_event_time = float('inf')
             return


        min_interval, max_interval = self.force_interval_range
        if not (0 <= min_interval <= max_interval):
             print(f"Warning: Invalid force interval range {self.force_interval_range}. Disabling forces.")
             self._next_force_event_time = float('inf')
             return

        min_duration, _ = self.force_duration_range
        safe_min_interval = max(min_interval, min_duration + 1e-6)

        # Use self.np_random directly
        next_interval = self.np_random.uniform(safe_min_interval, max_interval)
        self._next_force_event_time = current_time + next_interval

    def _start_random_force(self, current_time: float):
        """Samples and initiates a new random force application."""
        if self._force_body_id is None:
            return
        if self.np_random is None: # Check RNG availability
             print("Warning: np_random not available in _start_random_force. Cannot start force.")
             return

        force_vec = np.zeros(3)
        torque_vec = np.zeros(3)

        # --- Sample Translational Force ---
        if self.apply_translational_forces:
            min_mag, max_mag = self.translational_force_magnitude_range
            if 0 <= min_mag <= max_mag:
                magnitude = self.np_random.uniform(min_mag, max_mag)
                direction = self.np_random.standard_normal(3)
                norm = np.linalg.norm(direction)
                if norm > 1e-9: direction /= norm
                else: direction = np.array([1.0, 0.0, 0.0])
                force_vec = direction * magnitude
            # else: print warning handled elsewhere or implicitly skipped

        # --- Sample Rotational Force (Torque) ---
        if self.apply_rotational_forces:
            min_mag, max_mag = self.rotational_force_magnitude_range
            if 0 <= min_mag <= max_mag:
                magnitude = self.np_random.uniform(min_mag, max_mag)
                axis = self.np_random.standard_normal(3)
                norm = np.linalg.norm(axis)
                if norm > 1e-9: axis /= norm
                else: axis = np.array([1.0, 0.0, 0.0])
                torque_vec = axis * magnitude
            # else: print warning handled elsewhere or implicitly skipped

        self._current_force_vector = np.concatenate([force_vec, torque_vec])

        # --- Sample Duration ---
        min_dur, max_dur = self.force_duration_range
        if not (0 <= min_dur <= max_dur):
             duration = max(0.0, min_dur)
        else:
             duration = self.np_random.uniform(min_dur, max_dur)

        self._force_end_time = current_time + duration
        # Schedule the *next* time a force *event* can start
        self._schedule_next_force_event(current_time)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and schedules the first force event."""
        # super().reset() handles seeding self.np_random
        obs, info = super().reset(seed=seed, options=options)

        # Reset internal force state
        self._current_force_vector.fill(0)
        self._force_end_time = float('-inf')
        # Ensure np_random is available after reset before scheduling
        if self.np_random is None:
             print("Warning: np_random is None after reset. Cannot schedule initial force.")
             self._next_force_event_time = float('inf')
        else:
            current_time = self.env.data.time # Should be 0.0
            if self.apply_at_reset:
                 self._start_random_force(current_time) # Start one immediately
                 # _start_random_force already schedules the next event
            else:
                 self._schedule_next_force_event(current_time) # Schedule the first potential event

        # Update info dictionary
        info["next_force_event_time"] = self._next_force_event_time
        info["force_end_time"] = self._force_end_time
        info["is_force_active"] = self.env.data.time < self._force_end_time # Check initial state
        info["current_force_vector"] = self._current_force_vector.copy()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps the environment, applying forces as scheduled."""
        if self._force_body_id is None:
            return self.env.step(action) # Passthrough if disabled

        # Ensure np_random is available (it should be after reset)
        if self.np_random is None:
             print("Warning: np_random is None during step. Cannot manage forces.")
             # Apply zero force and step
             if hasattr(self.env.data, 'xfrc_applied'):
                 self.env.data.xfrc_applied[self._force_body_id] = 0.0
             return self.env.step(action)


        current_time = self.env.data.time

        # --- Manage Force State ---
        if current_time >= self._next_force_event_time:
            self._start_random_force(current_time)

        if current_time >= self._force_end_time and not np.all(self._current_force_vector == 0):
            self._current_force_vector.fill(0)
            self._force_end_time = float('-inf')

        # --- Apply Force ---
        try:
            # Check if xfrc_applied exists and has the right shape
            if hasattr(self.env.data, 'xfrc_applied') and \
               self.env.data.xfrc_applied.shape[0] > self._force_body_id:
                 self.env.data.xfrc_applied[self._force_body_id] = self._current_force_vector
            else:
                 # This case should ideally not happen if body ID is valid, but safety check
                 if self._force_body_id is not None: # Avoid repeated warnings
                      print(f"Warning: Cannot apply force to body ID {self._force_body_id}. xfrc_applied might be missing or too small.")
                      self._force_body_id = None # Disable to prevent further errors
                 self._current_force_vector.fill(0)

        except Exception as e: # Catch potential errors during assignment
             print(f"Error applying force via xfrc_applied: {e}. Disabling forces.")
             self._force_body_id = None
             self._current_force_vector.fill(0)


        # --- Step the Underlying Environment ---
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- Update Info ---
        info["next_force_event_time"] = self._next_force_event_time
        info["force_end_time"] = self._force_end_time
        info["is_force_active"] = self.env.data.time < self._force_end_time
        info["current_force_vector"] = self._current_force_vector.copy()

        return obs, reward, terminated, truncated, info
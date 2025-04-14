import csv
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class RewardCallback(BaseCallback):
    # Remove 'keys' from __init__ arguments
    def __init__(self, output_folder, real_time_flag=True, verbose=0):
        super().__init__(verbose)
        self.keys = [] # Initialize as empty, will be populated in _on_init
        self.data = {} # Initialize as empty, will be populated in _on_init
        self.column_order = [] # Initialize as empty
        self.real_time_column = [] # Initialize as empty
        self.step_counter = 0
        self.output_folder = output_folder # Store output folder path
        self.csv_file = os.path.join(self.output_folder, 'logs/rewards_continuous.csv') # Use output_folder
        self.real_time_flag = real_time_flag
        self._initialized = False # Flag to ensure initialization happens once

    def _on_init(self) -> None:
        """
        This method is called before the first rollout starts.
        Used here to get reward keys from the environment.
        """
        if not self._initialized:
            # Access the training environment
            # For VecEnv, get attributes from the underlying environments
            if isinstance(self.training_env, (SubprocVecEnv, DummyVecEnv)):
                # Get 'reward_keys' from the first underlying environment
                self.keys = self.training_env.get_attr('reward_keys')[0]
            else:
                # Handle non-vectorized environment case (if applicable)
                self.keys = self.training_env.reward_keys

            if not self.keys:
                 raise ValueError("Could not retrieve reward_keys from the environment.")

            # Now initialize data structures based on the retrieved keys
            self.data = {
                'rewards': [],
                'std': [],
                'components': {key: [] for key in self.keys}
            }
            self.column_order = ['Training Steps'] + self.keys + ['Reward', 'Std', 'Condition']
            self.real_time_column = ['Training Steps'] + self.keys + ['Reward']

            # Initialize CSV file if real-time logging is enabled
            if self.real_time_flag:
                # Ensure the logs directory exists
                os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
                if os.path.exists(self.csv_file):
                    print(f"Removing existing continuous log file: {self.csv_file}")
                    os.remove(self.csv_file)

                with open(self.csv_file, 'w', newline='') as f: # Use newline='' for csv writer
                    writer = csv.writer(f)
                    writer.writerow(self.real_time_column)
                print(f"Initialized continuous log file: {self.csv_file}")

            self._initialized = True # Mark as initialized

    def _initialize_on_step(self) -> bool:
        """
        Attempt initialization from _on_step if _on_init failed or was skipped.
        Returns True if initialization is now complete, False otherwise.
        """
        if self.verbose > 0:
            print("RewardCallback: Attempting initialization from _on_step...")
        # Re-run the initialization logic (which includes the check)
        self._on_init()
        return self._initialized # Return the current status

    def _on_step(self) -> bool:
        
        # If not initialized, try to initialize. If it fails again, skip this step.
        if not self._initialized:
            # Try initializing again. If it fails, print a more persistent warning once.
            if not self._initialize_on_step():
                 if not hasattr(self, '_init_failed_warning_printed'): # Print only once
                     print("Warning: RewardCallback initialization failed. Skipping step processing until initialized.")
                     self._init_failed_warning_printed = True
                 return True # Continue training, maybe it initializes later
            
        # Check if 'infos' is available and not empty
        if "infos" not in self.locals or not self.locals["infos"]:
            # This can happen at the very beginning or under specific VecEnv conditions
            # print("Warning: 'infos' not found or empty in callback locals.")
            return True # Continue training

        infos = self.locals["infos"]
        # print(f"Infos: {infos}") # Keep for debugging if needed

        # --- Access reward components directly from info dict ---
        # The keys should now be top-level in each info dictionary
        current_components = {}
        valid_infos_count = 0
        for key in self.keys:
            component_values = []
            for info in infos:
                # Check if the key exists in the specific info dict
                if key in info:
                    component_values.append(info[key])
                # else:
                #     print(f"Warning: Key '{key}' not found in info dict: {info}") # Debugging
            if component_values: # Only calculate mean if we found the key
                 current_components[key] = np.mean(component_values)
                 valid_infos_count = max(valid_infos_count, len(component_values)) # Track how many infos had keys
            # else:
                 # Handle cases where a key might be missing entirely across all infos
                 # current_components[key] = 0.0 # Or np.nan, or raise an error

        # Check if we processed any valid info dicts for components
        if valid_infos_count == 0 and self.keys:
             # print("Warning: No reward component keys found in any 'infos' dictionary this step.")
             # Decide how to handle this: skip logging, log zeros/NaNs, or error
             # For now, let's just skip appending component data if none were found
             pass # Or set components to default values if needed
        else:
            # Append data only if components were successfully extracted
            self.data['rewards'].append(np.mean(self.locals["rewards"]))
            self.data['std'].append(np.std(self.locals["rewards"]))
            for key in self.keys:
                 # Use .get() with a default if a key might be missing from current_components
                 self.data['components'][key].append(current_components.get(key, np.nan)) # Use NaN for missing keys

            # Real-time logging
            if self.real_time_flag:
                row_data = {
                    'Training Steps': self.num_timesteps, # Use self.num_timesteps for total steps
                    'Reward': self.data['rewards'][-1],
                }
                # Add components, using NaN for any potentially missing ones
                row_data.update({k: current_components.get(k, np.nan) for k in self.keys})
                # self.step_counter += 1 # self.num_timesteps is better

                try:
                    with open(self.csv_file, 'a', newline='') as f: # Use newline=''
                        # Ensure fieldnames match the actual keys being written
                        writer = csv.DictWriter(f, fieldnames=self.real_time_column)
                        # Filter row_data to only include keys present in real_time_column
                        filtered_row_data = {k: row_data[k] for k in self.real_time_column if k in row_data}
                        writer.writerow(filtered_row_data)
                except Exception as e:
                    print(f"Error writing to CSV {self.csv_file}: {e}")
                    print(f"Row data: {row_data}")
                    print(f"Fieldnames: {self.real_time_column}")


        return True # Continue training

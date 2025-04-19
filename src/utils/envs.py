import gymnasium as gym
import numpy as np
from typing import Optional

def find_wrapper_by_name(env: gym.Env, class_name_str: str) -> Optional[gym.Wrapper]:
    """
    Searches the wrapper stack for a wrapper whose class name contains the given string.

    Args:
        env: The environment instance (potentially wrapped).
        class_name_str: The string to look for in the wrapper class names.

    Returns:
        The wrapper instance if found, otherwise None.
    """
    current_env = env
    while True:
        # Check if the current environment's class name contains the target string
        if class_name_str in current_env.__class__.__name__:
            return current_env
        # If it's a wrapper, move to the next inner environment
        elif isinstance(current_env, gym.Wrapper):
            current_env = current_env.env
        # If it's not a wrapper and we haven't found the target, it's not in the stack
        else:
            return None # Wrapper not found

# --- Alternative using the 'class_name' attribute we added earlier ---
# This version is slightly more robust if you consistently add the class_name attribute
# def find_wrapper_by_class_name_attr(env: gym.Env, target_class_name: str) -> Optional[gym.Wrapper]:
#     """Searches using the 'class_name' attribute."""
#     current_env = env
#     while True:
#         if hasattr(current_env, 'class_name') and current_env.class_name == target_class_name:
#             return current_env
#         elif isinstance(current_env, gym.Wrapper):
#             current_env = current_env.env
#         else:
#             return None

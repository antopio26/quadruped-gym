# src/env_builder.py
import numpy as np
from typing import Optional, Dict, Any, Tuple, Type

import gymnasium as gym

from src.envs.base_quad import QuadrupedEnv
from src.envs.wrappers.control_input import ControlInputWrapper
from src.envs.wrappers.partial_observation import PartialObservationWrapper
from src.envs.wrappers.sequenced_rewards import SequencedRewardWrapper
from src.envs.wrappers.velocity_action import VelocityActionWrapper
# Import the new wrapper
from src.envs.wrappers.random_force_disturb import RandomForceDisturbWrapper
from src.controls.velocity_heading_controls import VelocityHeadingControls

def create_quadruped_env(
    # --- Base Env Args ---
    model_path: str = "./models/quadruped/scene.xml",
    max_time: float = 10.0,
    frame_skip: int = 10,
    render_mode: Optional[str] = None,
    width: int = 720,
    height: int = 480,
    render_fps: int = 30,
    save_video: bool = False,
    video_path: str = "videos/simulation.mp4",
    reset_options: Optional[Dict[str, Any]] = None,
    # --- Wrapper Args ---
    # Velocity Action Wrapper
    use_velocity_wrapper: bool = True,
    max_joint_speed: float = 5.0, # Max speed for velocity wrapper (unit/s)
    # Control Input Wrapper
    control_logic_class: Type = VelocityHeadingControls, # Allow specifying control logic class
    control_kwargs: Optional[Dict[str, Any]] = None, # Args for control logic constructor
    # Partial Observation Wrapper
    add_po_wrapper: bool = True,
    obs_window: int = 1,
    # Reward Wrapper
    add_reward_wrapper: bool = True,
    # Random Force Wrapper Args
    add_force_wrapper: bool = False, # Default to False
    random_force_options: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    """
    Builds the wrapped Quadruped environment stack, including optional
    random impulse disturbances.

    Args:
        # BaseQuadrupedEnv args...
        # VelocityActionWrapper args...
        # ControlInputWrapper args...
        # PartialObservationWrapper args...
        # SequencedRewardWrapper args...
        # RandomImpulseDisturbWrapper args...
        add_impulse_wrapper: If True, adds the RandomImpulseDisturbWrapper.

    Returns:
        The fully wrapped Gymnasium environment.
    """
    if control_kwargs is None:
        control_kwargs = {}

    # 1. Base Environment
    env = QuadrupedEnv(
        model_path=model_path,
        max_time=max_time,
        frame_skip=frame_skip,
        render_mode=render_mode,
        width=width,
        height=height,
        render_fps=render_fps,
        save_video=save_video,
        video_path=video_path,
        reset_options=reset_options # Pass reset options here
    )

    # 2. Random Impulse Wrapper (Optional - applied early)
    if add_force_wrapper:
        env = RandomForceDisturbWrapper(
            env=env,
            **random_force_options
        )

    # 3. Velocity Action Wrapper (Optional - applied before ControlInputWrapper)
    if use_velocity_wrapper:
        env = VelocityActionWrapper(
            env=env,
            max_speed=max_joint_speed
        )

    # 4. Control Input Wrapper
    env = ControlInputWrapper(
        env=env,
        control_logic_class=control_logic_class,
        **control_kwargs
    )

    # 5. Partial Observation Wrapper (Optional)
    if add_po_wrapper:
        env = PartialObservationWrapper(
            env=env,
            obs_window=obs_window,
            # expected_control_obs_size is inferred from ControlInputWrapper
        )

    # 6. Walking Reward Wrapper (Optional)
    if add_reward_wrapper:
        env = SequencedRewardWrapper(
            env=env
            # Add any reward wrapper specific args here if needed
        )

    return env
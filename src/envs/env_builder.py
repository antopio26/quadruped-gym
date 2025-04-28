import numpy as np
from typing import Optional, Dict, Any

import gymnasium as gym

from src.envs.base_quad import QuadrupedEnv
from src.envs.wrappers.control_input import ControlInputWrapper
from src.envs.wrappers.partial_observation import PartialObservationWrapper
from src.envs.wrappers.sequenced_rewards import SequencesRewardWrapper
from src.envs.wrappers.velocity_action import VelocityActionWrapper
from src.controls.velocity_heading_controls import VelocityHeadingControls

def create_quadruped_env(
    model_path: str = "./models/quadruped/scene.xml",
    max_time: float = 10.0,
    frame_skip: int = 16,
    render_mode: Optional[str] = None,
    width: int = 720,
    height: int = 480,
    render_fps: int = 30,
    save_video: bool = False,
    video_path: str = "videos/simulation.mp4",
    reset_options: Optional[Dict[str, Any]] = None,
    # Wrapper specific args
    obs_window: int = 1,
    control_logic_class = VelocityHeadingControls, # Allow specifying control logic
    control_kwargs: Optional[Dict[str, Any]] = None, # Args for control logic constructor
    add_reward_wrapper: bool = True, # Option to skip reward wrapper if not needed
    add_po_wrapper: bool = True, # Option to skip PO wrapper if not needed
    use_velocity_wrapper: bool = True, # Option to use the velocity wrapper
    max_joint_speed: float = 20, # Max speed for velocity wrapper (unit/s)
) -> gym.Env:
    """
    Builds the wrapped Quadruped environment stack.

    Args:
        # BaseQuadrupedEnv args... (see above)
        # Wrapper args... (see above)
        control_logic_class: The class for control logic (e.g., VelocityHeadingControls).
        control_kwargs: Arguments to pass to the control_logic_class constructor.
        add_reward_wrapper: If True, adds the SequencedRewardWrapper.
        add_po_wrapper: If True, adds the PartialObservationWrapper.
        use_velocity_wrapper: If True, adds the VelocityActionWrapper.
        max_joint_speed: The maximum joint speed for the VelocityActionWrapper.

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

    # 2. Velocity Action Wrapper (Optional - applied before ControlInputWrapper)
    if use_velocity_wrapper:
        # Ensure the base env has the required method if using the first version
        # if not hasattr(env.unwrapped, 'get_joint_positions'):
        #     raise AttributeError("Base environment needs 'get_joint_positions' for the first VelocityActionWrapper version.")
        env = VelocityActionWrapper(
            env=env,
            max_speed=max_joint_speed
        )

    # 3. Control Input Wrapper
    env = ControlInputWrapper(
        env=env,
        control_logic_class=control_logic_class,
        **control_kwargs
    )

    # 4. Partial Observation Wrapper (Optional)
    if add_po_wrapper:
        env = PartialObservationWrapper(
            env=env,
            obs_window=obs_window,
            # expected_control_obs_size is inferred from ControlInputWrapper
        )

    # 5. Walking Reward Wrapper (Optional)
    if add_reward_wrapper:
        env = SequencesRewardWrapper(
            env=env
        )

    return env
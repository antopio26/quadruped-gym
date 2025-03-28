import numpy as np
from stable_baselines3 import PPO
from envs.po_walking_quad import WalkingQuadrupedEnv
import matplotlib.pyplot as plt

def evaluate_model(model_path):
    env = WalkingQuadrupedEnv(render_mode="human", render_fps=30, save_video=False, frame_window=5)

    env.control_inputs.set_orientation(np.pi / 2)
    env.control_inputs.set_velocity_speed_alpha(1.0, np.pi / 2)

    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    rewards = []

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        env.render()

    env.close()

    # Plot the rewards over steps
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Evaluation Rewards Over Steps')
    plt.show()

if __name__ == '__main__':
    model_path = '../policies/po/ppo_quadruped.zip'
    evaluate_model(model_path)
import numpy as np
from stable_baselines3 import PPO
from envs.po_walking_quad import POWalkingQuadrupedEnv
import matplotlib.pyplot as plt


def evaluate_model(model_path):
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    env = POWalkingQuadrupedEnv(obs_window=5, random_init=True)

    env.control_inputs.set_orientation(0)
    env.control_inputs.set_velocity_speed_alpha(0.2, 0)

    model = PPO.load(model_path, env=env)

    for i in range(8):
        obs, _ = env.reset()
        done = False
        rewards = []

        while not done:
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            env.render()

        # Plot the rewards over steps
        plt.plot(range(len(rewards)), rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Evaluation Rewards Over Steps')
        plt.show()

    env.close()

if __name__ == '__main__':
    model_path = '../policies/po_ppo_local_ideal_v1/policy.zip'
    evaluate_model(model_path)
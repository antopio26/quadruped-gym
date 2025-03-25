import numpy as np
import matplotlib.pyplot as plt
from envs.walking_quadruped import WalkingQuadrupedEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Instantiate the environment.
env = WalkingQuadrupedEnv(render_mode="human", render_fps=30, save_video=True)

model = A2C("MlpPolicy", env, verbose=1)

# Callback to record rewards
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals["rewards"])
        return True

reward_callback = RewardCallback()

# Train the model
model.learn(total_timesteps=100_000, callback=reward_callback)

# Plot the rewards over training steps
plt.plot(reward_callback.rewards)
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.title('Reward Over Training Steps')
plt.show()

# Evaluate the model
vec_env = model.get_env()
obs = vec_env.reset()

done = False
rewards = []

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    env.render()

env.close()

# Plot the rewards over time
plt.plot(rewards)
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.show()
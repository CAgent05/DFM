import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

model = DQN.load('dqn_mountaincar_15e5.zip')
env = gym.make('MountainCar-v0',
               render_mode='rgb_array')


cnt = 0

for i in range(100):
    obs, _ = env.reset()
    env.render()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if abs(obs[0]+1.2) <= 0.000001:
            cnt += 1
            print('Function Falut')
            break
        total_reward += reward
    print(total_reward)

print(cnt)

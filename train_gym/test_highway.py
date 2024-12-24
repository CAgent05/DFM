import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import highway_env


model = DQN.load('dqn_highway1-fast-8e4.zip')
env = gym.make(
    'highway-fast-v0',
    render_mode='human',
    config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 3,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "collision_reward": -10,
        "lanes_count": 3,
        "duration": 100,
    })

cnt = 0

for i in range(10):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    print(total_reward)
    if info['crashed']:
        cnt += 1

print(cnt)

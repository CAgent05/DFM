import gymnasium as gym
from stable_baselines3 import DQN, PPO, SAC
import torch
import numpy as np
import highway_env


def make_env_model(env_name):
    
    model_path = './gymmodel/' + env_name + '.zip'
    
    if env_name == 'CartPole':
        env = gym.make(env_name + '-v0')
        model = DQN.load(model_path)
    elif env_name == 'MountainCar':
        env = gym.make(env_name + '-v0')
        model = DQN.load(model_path)
    elif env_name == 'BipedalWalkerHC':
        env = gym.make('BipedalWalker-v3',
                    hardcore=True,
                    render_mode='rgb_array')
        model = SAC.load(model_path)
    elif env_name == 'Highway':
        env = gym.make(
                'highway-fast-v0',
                render_mode='rgb_array',
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
                }
            )
        model = DQN.load(model_path)
    else:
        env = gym.make(env_name+ '-v4')
        if env_name == 'InvertedDoublePendulum':
            model = PPO.load(model_path)
        else:
            model = SAC.load(model_path)

    return env, model


def collect(env_name, n, total_reward, record, timeseries, labels, info=None, cnt=1000):
    
    if env_name == 'BipedalWalkerHC':
        if total_reward < 285:
            timeseries.append(torch.stack(record[-n:], dim=1))
            labels.append(1)
        else:
            index = np.random.randint(0, len(record)-n)
            timeseries.append(torch.stack(record[index:index+n], dim=1))
            labels.append(0)
    elif env_name == 'CartPole':
        if total_reward < 200:
            timeseries.append(torch.stack(record[-n:], dim=1))
            labels.append(1)
        else:
            index = np.random.randint(0, len(record)-n)
            timeseries.append(torch.stack(record[index:index+n], dim=1))
            labels.append(0)
    elif env_name == 'MountainCar':
        if total_reward == -200:
            timeseries.append(torch.stack(record[-n:], dim=1))
            labels.append(1)
        else:
            index = np.random.randint(0, len(record)-n)
            timeseries.append(torch.stack(record[index:index+n], dim=1))
            labels.append(0)
    elif env_name == 'Highway':
        if len(record) < n:
            return
        if info['crashed']:
            timeseries.append(torch.stack(record[-n:], dim=1))
            labels.append(1)
        else:
            index = np.random.randint(0, len(record)-n)
            timeseries.append(torch.stack(record[index:index+n], dim=1))
            labels.append(0)
    else:
        if cnt < 1000:
            timeseries.append(torch.stack(record[-n:], dim=1))
            labels.append(1)
        else:
            index = np.random.randint(0, len(record)-n)
            timeseries.append(torch.stack(record[index:index+n], dim=1))
            labels.append(0)
    
    
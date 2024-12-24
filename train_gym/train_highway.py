import gymnasium as gym
from stable_baselines3 import DQN 
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env
import pprint

env = gym.make(
    'highway-fast-v0',
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
        "duration": 30,
    })

pprint.pprint(env.unwrapped.config)

model = DQN('MlpPolicy',
            env=env,
            verbose=1,
            exploration_final_eps=0.05,
            tensorboard_log='./train_gym/tensorboardlog/highway/')

model.learn(total_timesteps=9e4)

model.save("./dqn_highway1-fast-9e4")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

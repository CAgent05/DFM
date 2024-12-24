import gymnasium as gym
from stable_baselines3 import DQN 
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env

env = gym.make('MountainCar-v0')


model = DQN('MlpPolicy',
            env=env,
            verbose=1,
            exploration_final_eps=0.05,
            tensorboard_log='./train_gym/tensorboardlog/mountaincar/')

model.learn(total_timesteps=1500000)

model.save("./dqn_mountaincar_15e5")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

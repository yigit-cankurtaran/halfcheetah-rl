from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def watch(
    env_path="./env/vecnorm.pkl",
    model_path="./model/best_model.zip",
    env_name="HalfCheetah-v5",
    timesteps=5,
):
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name, render_mode="human"))])
    env = VecNormalize.load(env_path, env)
    env.norm_reward = False
    env.training = False

    model = PPO.load(model_path, env)

    rewards, lengths = evaluate_policy(
        model, env, timesteps, render=True, return_episode_rewards=True
    )

    for i in range(len(rewards)):
        print(f"run={i + 1}, reward={rewards[i]}, length={lengths[i]}")

    env.close()


if __name__ == "__main__":
    watch()

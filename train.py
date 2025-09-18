import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os


def linear_decay(init_val):
    def returned(progress_remaining):
        return init_val * progress_remaining

    return returned


def train(timesteps=1_000_000):
    paths = ["model", "logs", "env"]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    model_path, logs_path, _ = paths

    env_path = "./env/vecnorm.pkl"

    train_env = VecNormalize(make_vec_env("HalfCheetah-v5", 4))
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(gym.make("HalfCheetah-v5"))]))

    eval_call = EvalCallback(
        eval_env,
        log_path=logs_path,
        best_model_save_path=model_path,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=0.995,  # long locomotion task, we need higher consideration for later rewards
        ent_coef=0.02,
        n_steps=4096,
        learning_rate=linear_decay(1e-5),  # decaying as it goes on
    )

    model.learn(timesteps, eval_call, progress_bar=True)
    train_env.save(env_path)


if __name__ == "__main__":
    train()

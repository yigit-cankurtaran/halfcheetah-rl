import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os


class SaveEnvOnNewBest(BaseCallback):
    def __init__(self, train_env, env_path, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.env_path = env_path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # This will be called by EvalCallback when a new best model is found
        if hasattr(self, 'parent') and hasattr(self.parent, 'last_mean_reward'):
            self.train_env.save(self.env_path)


def linear_decay(init_val):
    def returned(progress_remaining):
        return init_val * progress_remaining

    return returned


def train(timesteps=1_000_000):
    paths = ["model", "logs", "env"]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    model_path, logs_path, env_path = paths

    train_env = VecNormalize(make_vec_env("HalfCheetah-v5", 4))
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(gym.make("HalfCheetah-v5"))]))

    save_env_callback = SaveEnvOnNewBest(train_env, env_path)

    eval_call = EvalCallback(
        eval_env,
        log_path=logs_path,
        best_model_save_path=model_path,
        callback_on_new_best=save_env_callback,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=0.995,  # long locomotion task, we need higher consideration for later rewards
        n_steps=4096,
        learning_rate=linear_decay(1e-5),  # decaying as it goes on
    )

    model.learn(timesteps, eval_call, progress_bar=True)


if __name__ == "__main__":
    train()

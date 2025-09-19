import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import argparse
import logging
import json
import torch


def setup_logging(log_level="INFO"):
    """setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def linear_decay(init_val):
    """linear learning rate decay schedule"""

    def returned(progress_remaining):
        return init_val * progress_remaining

    return returned


def load_config(config_path="config.json"):
    """load training configuration from json file"""
    default_config = {
        "env_name": "HalfCheetah-v5",
        "n_envs": 4,
        "timesteps": 5_000_000,
        "gamma": 0.995,
        "ent_coef": 0.02,
        "n_steps": 4096,
        "learning_rate": 1e-3,
        "seed": 42,
        "eval_freq": 10000,
        "n_eval_episodes": 5,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            # merge with defaults for missing keys
            default_config.update(config)
            return default_config
        except Exception as e:
            logging.warning(f"failed to load config from {config_path}: {e}")
            logging.info("using default configuration")
    else:
        logging.info(f"config file {config_path} not found, using defaults")

    return default_config


def create_directories(paths):
    """create necessary directories with error handling"""
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"created/verified directory: {path}")
        except Exception as e:
            logging.error(f"failed to create directory {path}: {e}")
            raise


def save_config(config, path="logs/training_config.json"):
    """save training configuration for reproducibility"""
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logging.info(f"saved training config to {path}")
    except Exception as e:
        logging.warning(f"failed to save config: {e}")


def train(config_path="config.json", **kwargs):
    """train ppo agent on halfcheetah environment"""
    logger = setup_logging()

    # load configuration
    config = load_config(config_path)
    # override with any kwargs
    config.update(kwargs)

    logger.info("starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # check for gpu availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    # create directories
    paths = ["model", "logs", "env"]
    create_directories(paths)
    model_path, logs_path, env_dir = paths

    # save config for reproducibility
    save_config(config, f"{logs_path}/training_config.json")

    env_path = f"{env_dir}/vecnorm.pkl"

    try:
        # create training environment
        train_env = VecNormalize(
            make_vec_env(config["env_name"], config["n_envs"], seed=config["seed"])
        )

        # create evaluation environment (shares normalization but doesn't update it)
        eval_env = VecNormalize(
            DummyVecEnv([lambda: Monitor(gym.make(config["env_name"]))]),
            training=False,  # important: don't update normalization during eval
        )

        logger.info("created training and evaluation environments")

        # setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            log_path=logs_path,
            best_model_save_path=model_path,
            eval_freq=config["eval_freq"],
            n_eval_episodes=config["n_eval_episodes"],
            deterministic=True,
            render=False,
        )

        # create model
        model = PPO(
            "MlpPolicy",
            train_env,
            gamma=config["gamma"],
            ent_coef=config["ent_coef"],
            n_steps=config["n_steps"],
            learning_rate=linear_decay(config["learning_rate"]),
            device=device,
            seed=config["seed"],
            verbose=1,
        )

        logger.info("created ppo model")
        logger.info(f"model parameters: {model.get_parameters()}")

        # start training
        logger.info(f"starting training for {config['timesteps']} timesteps")
        model.learn(
            total_timesteps=config["timesteps"],
            callback=eval_callback,
            progress_bar=True,
        )

        # save environment normalization
        train_env.save(env_path)
        logger.info(f"saved environment normalization to {env_path}")

        # copy normalization to eval env for consistency
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

        logger.info("training completed successfully")

    except Exception as e:
        logger.error(f"training failed: {e}")
        raise
    finally:
        # cleanup
        if "train_env" in locals():
            train_env.close()
        if "eval_env" in locals():
            eval_env.close()


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="train ppo on halfcheetah")
    parser.add_argument("--config", default="config.json", help="path to config file")
    parser.add_argument("--timesteps", type=int, help="training timesteps")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--learning-rate", type=float, help="learning rate")
    parser.add_argument("--gamma", type=float, help="discount factor")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # convert args to dict, removing None values
    kwargs = {
        k.replace("-", "_"): v
        for k, v in vars(args).items()
        if v is not None and k != "config"
    }

    train(config_path=args.config, **kwargs)

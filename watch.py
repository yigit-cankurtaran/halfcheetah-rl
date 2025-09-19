from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
import argparse
import logging
import os
import json


def setup_logging(log_level="INFO"):
    """setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def validate_paths(env_path, model_path):
    """validate that required files exist"""
    logger = logging.getLogger(__name__)

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"environment normalization file not found: {env_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")

    logger.info(f"validated paths: env={env_path}, model={model_path}")


def compute_statistics(rewards, lengths):
    """compute and return evaluation statistics"""
    rewards_array = np.array(rewards)
    lengths_array = np.array(lengths)

    stats = {
        "n_episodes": len(rewards),
        "reward_mean": np.mean(rewards_array),
        "reward_std": np.std(rewards_array),
        "reward_min": np.min(rewards_array),
        "reward_max": np.max(rewards_array),
        "length_mean": np.mean(lengths_array),
        "length_std": np.std(lengths_array),
        "length_min": np.min(lengths_array),
        "length_max": np.max(lengths_array),
    }

    return stats


def save_results(rewards, lengths, stats, output_path=None):
    """save evaluation results to json file"""
    if output_path is None:
        return

    logger = logging.getLogger(__name__)

    results = {
        "statistics": stats,
        "episodes": [
            {"episode": i + 1, "reward": float(reward), "length": int(length)}
            for i, (reward, length) in enumerate(zip(rewards, lengths))
        ],
    }

    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"saved evaluation results to {output_path}")
    except Exception as e:
        logger.warning(f"failed to save results: {e}")


def watch(
    env_path="./env/vecnorm.pkl",
    model_path="./model/best_model.zip",
    env_name="HalfCheetah-v5",
    n_episodes=5,
    render=True,
    deterministic=True,
    output_path=None,
    seed=None,
):
    """evaluate trained model on environment"""
    logger = setup_logging()

    logger.info("starting model evaluation")
    logger.info(f"environment: {env_name}")
    logger.info(f"episodes: {n_episodes}")
    logger.info(f"deterministic: {deterministic}")
    logger.info(f"render: {render}")

    try:
        # validate file paths
        validate_paths(env_path, model_path)

        # create environment
        render_mode = "human" if render else None
        env = DummyVecEnv(
            [lambda: Monitor(gym.make(env_name, render_mode=render_mode))]
        )

        # load environment normalization
        try:
            env = VecNormalize.load(env_path, env)
            env.norm_reward = False  # don't normalize rewards during evaluation
            env.training = False  # disable normalization updates
            logger.info("loaded environment normalization")
        except Exception as e:
            logger.error(f"failed to load environment normalization: {e}")
            raise

        # load model
        try:
            model = PPO.load(model_path, env)
            logger.info("loaded trained model")
        except Exception as e:
            logger.error(f"failed to load model: {e}")
            raise

        # validate environment compatibility
        if hasattr(env, "observation_space") and hasattr(
            model.policy, "observation_space"
        ):
            env_obs_shape = env.observation_space.shape
            model_obs_shape = model.policy.observation_space.shape
            if env_obs_shape != model_obs_shape:
                logger.warning(
                    f"observation space mismatch: env={env_obs_shape}, "
                    f"model={model_obs_shape}"
                )

        # set seed if provided
        if seed is not None:
            env.seed(seed)
            logger.info(f"set random seed: {seed}")

        # evaluate policy
        logger.info("starting policy evaluation")
        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            render=render,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

        # compute statistics
        stats = compute_statistics(rewards, lengths)

        # display results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        for i in range(len(rewards)):
            print(
                f"episode {i + 1:2d}: reward={rewards[i]:8.2f}, length={lengths[i]:4d}"
            )

        print("\n" + "-" * 50)
        print("STATISTICS")
        print("-" * 50)
        print(f"episodes:     {stats['n_episodes']}")
        print(f"reward mean:  {stats['reward_mean']:8.2f} ± {stats['reward_std']:6.2f}")
        print(f"reward range: {stats['reward_min']:8.2f} to {stats['reward_max']:8.2f}")
        print(f"length mean:  {stats['length_mean']:8.1f} ± {stats['length_std']:6.1f}")
        print(f"length range: {stats['length_min']:4.0f} to {stats['length_max']:4.0f}")

        # confidence interval for mean reward (95%)
        if len(rewards) > 1:
            se = stats["reward_std"] / np.sqrt(stats["n_episodes"])
            ci_95 = 1.96 * se
            print(f"reward 95% ci: {stats['reward_mean']:8.2f} ± {ci_95:6.2f}")

        print("=" * 50)

        # save results if requested
        save_results(rewards, lengths, stats, output_path)

        logger.info("evaluation completed successfully")

        return rewards, lengths, stats

    except Exception as e:
        logger.error(f"evaluation failed: {e}")
        raise
    finally:
        # cleanup
        if "env" in locals():
            env.close()


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="evaluate trained ppo model")
    parser.add_argument(
        "--env-path",
        default="./env/vecnorm.pkl",
        help="path to environment normalization file",
    )
    parser.add_argument(
        "--model-path", default="./model/best_model.zip", help="path to trained model"
    )
    parser.add_argument(
        "--env-name", default="HalfCheetah-v5", help="gymnasium environment name"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5, help="number of evaluation episodes"
    )
    parser.add_argument("--no-render", action="store_true", help="disable rendering")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="use stochastic policy instead of deterministic",
    )
    parser.add_argument(
        "--output", type=str, help="path to save evaluation results (json)"
    )
    parser.add_argument("--seed", type=int, help="random seed for evaluation")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    watch(
        env_path=args.env_path,
        model_path=args.model_path,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        output_path=args.output,
        seed=args.seed,
    )

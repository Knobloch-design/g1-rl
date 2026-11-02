"""Train a PPO walking policy on G1Walk-v0 with Stable-Baselines3.

.venv/bin/python examples/train_ppo.py --num-envs 10 --steps 20_000_000
.venv/bin/tensorboard --logdir runs
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import g1_walking  # noqa: F401  (registers G1Walk-v0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--run", default="ppo_g1")
    # Looser than the env's 1e-3 default: ~3x faster early in training,
    # when the robot is mostly falling.
    parser.add_argument("--accuracy", type=float, default=1e-2)
    args = parser.parse_args()

    env = make_vec_env(
        "G1Walk-v0",
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(accuracy=args.accuracy),
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=1024,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[512, 256, 128], log_std_init=-1.0),
        tensorboard_log="runs",
        device="cpu",
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(500_000 // args.num_envs, 1),
        save_path=f"runs/{args.run}/checkpoints",
        save_vecnormalize=True,
    )
    model.learn(args.steps, callback=checkpoint, tb_log_name=args.run)
    model.save(f"runs/{args.run}/final")
    env.save(f"runs/{args.run}/vecnormalize.pkl")


if __name__ == "__main__":
    gym.logger.min_level = 40
    main()

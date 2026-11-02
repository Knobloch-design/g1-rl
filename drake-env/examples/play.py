"""Watch a policy in meshcat. With no --model, holds the home pose.

    .venv/bin/python examples/play.py
    .venv/bin/python examples/play.py --model runs/ppo_g1/final.zip \
        --vecnormalize runs/ppo_g1/vecnormalize.pkl --command 0.5 0 0
"""

import argparse
import pickle

import numpy as np

from g1_walking import G1WalkEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--vecnormalize")
    parser.add_argument("--command", type=float, nargs=3)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = G1WalkEnv(render_mode="human", command=args.command)

    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        obs_rms = None
        if args.vecnormalize:
            with open(args.vecnormalize, "rb") as f:
                vec = pickle.load(f)
            obs_rms, clip = vec.obs_rms, vec.clip_obs
            eps = vec.epsilon

        def policy(obs):
            if obs_rms is not None:
                obs = np.clip(
                    (obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                    -clip,
                    clip,
                )
            return model.predict(obs, deterministic=True)[0]
    else:

        def policy(obs):
            return np.zeros(env.action_space.shape)

    input("Open the meshcat URL above, then press Enter to start.")
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        total = 0.0
        for _ in range(1000):
            obs, reward, terminated, _, _ = env.step(policy(obs))
            total += reward
            if terminated:
                break
        print(
            f"episode {episode}: command {info['command'].round(2)}, "
            f"return {total:.2f}, t = {env.context.get_time():.2f} s"
        )


if __name__ == "__main__":
    main()

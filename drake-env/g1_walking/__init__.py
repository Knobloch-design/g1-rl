from gymnasium.envs.registration import register

from g1_walking.env import G1WalkEnv

register(
    id="G1Walk-v0",
    entry_point="g1_walking.env:G1WalkEnv",
    max_episode_steps=1000,  # 20 s at 50 Hz
)

__all__ = ["G1WalkEnv"]

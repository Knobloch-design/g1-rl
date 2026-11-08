"""Environment construction shared by ES training and evaluation."""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-Unitree-G1"


def make_train_env(num_envs: int, seed: int, device: str) -> ManagerBasedRlEnv:
    """The training task, with the command-speed curriculum turned off.

    The curriculum widens the commanded speeds after 120k env steps. PPO's
    3000-iteration run (72k steps per env) never got there, but ES runs full
    episodes every generation and would reach it within ~120 generations, so it
    is removed to keep every method on the same task. Everything else (sensor
    noise, pushes, friction / mass randomization, random commands) is unchanged.
    """
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum = {}
    return ManagerBasedRlEnv(cfg=cfg, device=device)

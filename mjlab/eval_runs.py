"""Evaluate saved policies from PPO and ES runs the same way, for comparison.

    .venv/bin/python eval_runs.py logs/rsl_rl/g1_velocity/<run> logs/es/g1_velocity/<run> ...

Each run's training curves use different numbers (PPO logs the return of its
noisy exploring policy, ES the return of its candidates over a shorter
horizon), so they can't be compared directly. Instead, every selected
checkpoint is replayed here, deterministically (no exploration noise) and
with fixed seeds, so every policy faces exactly the same robots, pushes and
commands:

  episodes  1024 robots, the training task (random commands, sensor noise,
            pushes, randomized friction / mass), one full 20 s episode each:
            mean return, fraction that never fall, tracking errors
  course    64 robots on render_progress.py's fixed 20 s course (stand, walk,
            turn, sidestep, walk back) without noise or pushes; a fallen robot
            respawns and the fall is counted

Writes <run>/eval.csv: one row per checkpoint, with the wall-clock time and
env steps at which that checkpoint was saved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from es.env import TASK, make_train_env
from es.policy import ESPolicy

# Same course as render_progress.py.
COURSE = [
    (0.0, [0.0, 0.0, 0.0]),
    (3.0, [0.8, 0.0, 0.0]),
    (8.0, [0.5, 0.0, 0.5]),
    (12.0, [0.0, 0.4, 0.0]),
    (15.0, [-0.5, 0.0, 0.0]),
    (18.0, [0.0, 0.0, 0.0]),
]
SEED = 1234
STEPS = 1000  # 20 s at 50 Hz


def course_command(t: float) -> list[float]:
    command = COURSE[0][1]
    for start, c in COURSE:
        if t >= start:
            command = c
    return command


def make_course_env(num_envs: int, device: str) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = SEED
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.init_velocity_prob = 0.0
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def pick(available: list[int], count: int) -> list[int]:
    """`count` checkpoints, denser early (where behaviour changes fastest)."""
    if len(available) <= count:
        return available
    targets = np.geomspace(max(1, available[1]), available[-1], count - 1)
    chosen = {available[0], available[-1]}
    for t in targets:
        remaining = [i for i in available if i not in chosen]
        if remaining and len(chosen) < count:
            chosen.add(min(remaining, key=lambda i: abs(i - t)))
    return sorted(chosen)


def load_yaml(path: Path) -> dict:
    # rsl_rl's saved configs contain !!python/tuple tags; these are our own files.
    return yaml.load(path.read_text(), Loader=yaml.UnsafeLoader)


class Run:
    """Checkpoints of one training run, PPO or ES, behind one interface."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.checkpoints = {
            int(re.fullmatch(r"model_(\d+)\.pt", p.name).group(1)): p
            for p in run_dir.glob("model_*.pt")
        }
        self.kind = "es" if (run_dir / "config.json").exists() else "ppo"
        if self.kind == "es":
            self.config = json.loads((run_dir / "config.json").read_text())
            hidden = self.config["hidden"]
            arch = "linear" if not hidden else "mlp" + "x".join(map(str, hidden))
            opt = self.config.get("optimizer")
            self.label = f"{self.config['algo']}_{arch}" + (f"_{opt}" if opt else "")
        else:
            self.agent_cfg = load_yaml(run_dir / "params" / "agent.yaml")
            env_text = (run_dir / "params" / "env.yaml").read_text()
            self.num_envs = int(re.search(r"^\s*num_envs:\s*(\d+)", env_text, re.M).group(1))
            hidden = self.agent_cfg["actor"]["hidden_dims"]
            opt = self.agent_cfg["algorithm"].get("optimizer", "adam")
            self.label = f"ppo_mlp{'x'.join(map(str, hidden))}_{opt}"
            self._wall_times = self._read_ppo_wall_times()

    def _read_ppo_wall_times(self) -> dict[int, float]:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        times: dict[int, float] = {}
        for events in sorted(self.dir.glob("events.out.tfevents.*")):
            acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
            acc.Reload()
            tags = acc.Tags()["scalars"]
            for tag in ("Perf/total_fps", "Loss/learning_rate", "Train/mean_reward"):
                if tag in tags:
                    for e in acc.Scalars(tag):
                        times.setdefault(e.step, e.wall_time)
                    break
        if not times:
            return {}
        # Wall time at iteration 0 ~ first logged time minus one iteration.
        steps = sorted(times)
        first = times[steps[0]]
        per_iter = (times[steps[-1]] - first) / max(1, steps[-1] - steps[0])
        start = first - per_iter * (steps[0] + 1)
        return {s: times[s] - start for s in steps}

    def info(self, ckpt: int) -> tuple[float, int]:
        """(wall-clock seconds, env steps) when checkpoint `ckpt` was saved."""
        if self.kind == "es":
            d = torch.load(self.checkpoints[ckpt], map_location="cpu", weights_only=False)
            return d["wall_time_s"], d["env_steps"]
        steps = ckpt * self.agent_cfg["num_steps_per_env"] * self.num_envs
        wall = self._wall_times.get(ckpt)
        if wall is None and self._wall_times:
            nearest = min(self._wall_times, key=lambda s: abs(s - ckpt))
            wall = self._wall_times[nearest]
        return (wall if wall is not None else float("nan")), steps

    def policy(self, ckpt: int, wrapped_env: RslRlVecEnvWrapper, device: str):
        if self.kind == "es":
            d = torch.load(self.checkpoints[ckpt], map_location=device, weights_only=False)
            return ESPolicy(d, device)
        runner = MjlabOnPolicyRunner(wrapped_env, copy.deepcopy(self.agent_cfg), device=device)
        runner.load(str(self.checkpoints[ckpt]), load_cfg={"actor": True},
                    strict=True, map_location=device)
        return runner.get_inference_policy(device=device)


@torch.inference_mode()
def eval_episodes(env: ManagerBasedRlEnv, policy) -> dict[str, float]:
    n = env.num_envs
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = TensorDict(env.reset()[0], batch_size=[n])
    robot, twist = env.scene["robot"], env.command_manager.get_term("twist")
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    ret = torch.zeros(n, device=env.device)
    length = torch.zeros(n, device=env.device)
    exy = torch.zeros(n, device=env.device)
    eyaw = torch.zeros(n, device=env.device)
    for _ in range(STEPS):
        o, rew, term, trunc, _ = env.step(policy(obs))
        obs = TensorDict(o, batch_size=[n])
        ret += rew * alive
        length += alive
        still = alive & ~(term | trunc)
        cmd = twist.vel_command_b
        exy += still * (cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]).norm(dim=1)
        eyaw += still * (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
        # A time-out at step 1000 is success, not a fall.
        alive = alive & ~term
        if not alive.any():
            break
    alive_steps = length.sum().clamp_min(1)
    return {
        "ep_return": ret.mean().item(),
        "ep_survival": alive.float().mean().item(),
        "ep_length": length.mean().item(),
        "ep_err_vel_xy": (exy.sum() / alive_steps).item(),
        "ep_err_vel_yaw": (eyaw.sum() / alive_steps).item(),
    }


@torch.inference_mode()
def eval_course(env: ManagerBasedRlEnv, policy) -> dict[str, float]:
    n = env.num_envs
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = TensorDict(env.reset()[0], batch_size=[n])
    robot, twist = env.scene["robot"], env.command_manager.get_term("twist")
    falls = torch.zeros(n, device=env.device)
    exy = torch.zeros(n, device=env.device)
    eyaw = torch.zeros(n, device=env.device)
    ok_steps = torch.zeros(n, device=env.device)
    dist = torch.zeros(n, device=env.device)
    for step in range(STEPS):
        twist.vel_command_b[:] = torch.tensor(course_command(step * env.step_dt),
                                              device=env.device)
        obs = TensorDict(env.observation_manager.compute(), batch_size=[n])
        o, _, term, trunc, _ = env.step(policy(obs))
        done = term | trunc
        falls += done
        up = ~done
        cmd = twist.vel_command_b
        lin = robot.data.root_link_lin_vel_b[:, :2]
        exy += up * (cmd[:, :2] - lin).norm(dim=1)
        eyaw += up * (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
        dist += up * lin.norm(dim=1) * env.step_dt
        ok_steps += up
    return {
        "course_falls": falls.mean().item(),
        "course_no_fall_frac": (falls == 0).float().mean().item(),
        "course_err_vel_xy": (exy.sum() / ok_steps.sum().clamp_min(1)).item(),
        "course_err_vel_yaw": (eyaw.sum() / ok_steps.sum().clamp_min(1)).item(),
        "course_distance_m": dist.mean().item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--count", type=int, default=12, help="checkpoints per run")
    p.add_argument("--episode-envs", type=int, default=1024)
    p.add_argument("--course-envs", type=int, default=64)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    device = "cuda:0"

    ep_env = make_train_env(args.episode_envs, SEED, device)
    course_env = make_course_env(args.course_envs, device)
    # rsl_rl's runner needs a wrapped env to build the networks; the wrapper
    # resets the env it wraps, which every evaluation does anyway.
    course_wrapped = RslRlVecEnvWrapper(course_env, clip_actions=load_rl_cfg(TASK).clip_actions)

    for run_dir in args.run_dirs:
        out = run_dir / "eval.csv"
        if args.skip_existing and out.exists():
            print(f"{run_dir.name}: eval.csv exists, skipping")
            continue
        run = Run(run_dir)
        if not run.checkpoints:
            print(f"{run_dir.name}: no checkpoints, skipping")
            continue
        chosen = pick(sorted(run.checkpoints), args.count)
        print(f"{run_dir.name} [{run.label}]: checkpoints {chosen}", flush=True)
        rows = []
        for ckpt in chosen:
            policy = run.policy(ckpt, course_wrapped, device)
            wall, steps = run.info(ckpt)
            row = {"run": run_dir.name, "label": run.label, "kind": run.kind,
                   "checkpoint": ckpt, "wall_time_s": wall, "env_steps": steps,
                   **eval_episodes(ep_env, policy), **eval_course(course_env, policy)}
            rows.append(row)
            print(f"  {ckpt:5d}  {wall / 60:6.1f} min  return {row['ep_return']:6.2f}  "
                  f"survive {row['ep_survival']:.2f}  course falls "
                  f"{row['course_falls']:.2f}  err {row['course_err_vel_xy']:.2f}",
                  flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

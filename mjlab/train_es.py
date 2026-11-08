"""Train the G1 walking policy with an evolutionary method instead of PPO.

    .venv/bin/python train_es.py --algo ars --hidden            # linear policy
    .venv/bin/python train_es.py --algo pgpe --hidden 32 32 --time-limit-min 20
    .venv/bin/python train_es.py --algo sep_cmaes --hidden 512 256 128

Algorithms: ars, openai_es, pgpe, cmaes (full covariance), sep_cmaes (diagonal).

One generation:
  1. the algorithm proposes `--popsize` parameter vectors,
  2. each is run for `--episodes-per-candidate` episodes in parallel
     (popsize x episodes robots), all starting from a fresh reset, for up to
     `--horizon` policy steps (1000 = the full 20 s episode). A robot stops
     earning reward once it falls,
  3. a candidate's fitness is its mean episode return,
  4. the algorithm updates its search distribution from the fitnesses.

`--center-envs` extra robots run the current mean policy (no perturbation)
each generation, so the logs show how good the policy itself is, not just
the noisy candidates.

Logs go to logs/es/g1_velocity/<date>_<algo>_<arch>/:
  events.out.tfevents.*  TensorBoard
  progress.csv           one row per generation (same numbers)
  model_<gen>.pt         the mean policy + observation normalizer
  config.json            all settings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from es.algorithms import make_algorithm
from es.env import make_train_env
from es.policy import FlatMLP, ObsNormalizer

ALGOS = ["ars", "openai_es", "pgpe", "cmaes", "sep_cmaes"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--hidden", type=int, nargs="*", default=[32, 32],
                   help="hidden layer sizes; pass nothing for a linear policy")
    p.add_argument("--popsize", type=int, default=2048,
                   help="candidates per generation (even for antithetic methods)")
    p.add_argument("--episodes-per-candidate", type=int, default=2)
    p.add_argument("--center-envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=1000,
                   help="max policy steps per episode (50 per second)")
    p.add_argument("--generations", type=int, default=100000)
    p.add_argument("--time-limit-min", type=float, default=20.0)
    p.add_argument("--sigma", type=float, default=None,
                   help="exploration noise (ARS/OpenAI-ES), initial sigma "
                        "(PGPE per-parameter, CMA-ES global)")
    p.add_argument("--lr", type=float, default=None,
                   help="step size (ARS), Adam lr (OpenAI-ES), "
                        "ClipUp max speed (PGPE); unused by CMA-ES")
    p.add_argument("--optimizer", choices=["adam", "sgd", "clipup"], default=None,
                   help="how OpenAI-ES / PGPE apply their gradient estimate "
                        "(defaults: adam for OpenAI-ES, clipup for PGPE)")
    p.add_argument("--top-dirs", type=int, default=None, help="ARS: b")
    p.add_argument("--weight-decay", type=float, default=0.005,
                   help="OpenAI-ES L2 coefficient")
    p.add_argument("--action-clip", type=float, default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="")
    p.add_argument("--log-root", type=Path, default=Path("logs/es/g1_velocity"))
    args = p.parse_args()
    if args.algo in ("ars", "openai_es", "pgpe") and args.popsize % 2:
        p.error("antithetic methods need an even --popsize")
    return args


def arch_name(hidden) -> str:
    return "linear" if not hidden else "mlp" + "x".join(map(str, hidden))


def main():
    args = parse_args()
    device = "cuda:0"
    torch.manual_seed(args.seed)

    K = args.episodes_per_candidate
    P = args.popsize
    m = max(1, math.ceil(args.center_envs / K))  # center copies, K robots each
    num_envs = (P + m) * K
    env = make_train_env(num_envs, args.seed, device)
    obs_dim = env.observation_manager.group_obs_dim["actor"][0]
    act_dim = env.action_manager.total_action_dim
    robot = env.scene["robot"]
    twist = env.command_manager.get_term("twist")

    net = FlatMLP(obs_dim, act_dim, tuple(args.hidden))
    theta0 = net.init_params(torch.Generator().manual_seed(args.seed)).to(device)
    algo = make_algorithm(args.algo, theta0, args)
    assert algo.popsize == P, (algo.popsize, P)
    normalizer = ObsNormalizer(obs_dim, device)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{args.algo}_{arch_name(args.hidden)}"
    if args.optimizer:
        name += f"_{args.optimizer}"
    if args.run_name:
        name += f"_{args.run_name}"
    log_dir = args.log_root / name
    log_dir.mkdir(parents=True)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "obs_dim": obs_dim, "act_dim": act_dim, "num_params": net.num_params,
              "num_envs": num_envs, "center_copies": m}
    (log_dir / "config.json").write_text(json.dumps(config, indent=2))
    writer = SummaryWriter(str(log_dir))
    csv_file = open(log_dir / "progress.csv", "w", newline="")
    csv_writer = None
    print(f"{name}: {net.num_params:,} parameters, popsize {P} x {K} episodes "
          f"+ {m * K} center robots = {num_envs} envs")

    is_center = torch.zeros(num_envs, dtype=torch.bool, device=device)
    is_center[P * K :] = True

    def save(gen, env_steps, wall):
        torch.save({"theta": algo.mean.cpu(), "obs_norm": normalizer.state_dict(),
                    "config": config, "generation": gen, "env_steps": env_steps,
                    "wall_time_s": wall}, log_dir / f"model_{gen:05d}.pt")

    start = time.time()
    env_steps = 0
    alive_steps = 0
    save(0, 0, 0.0)
    for gen in range(1, args.generations + 1):
        t0 = time.time()
        with torch.no_grad():
            cand = algo.ask()
            params = torch.cat([cand, algo.mean.unsqueeze(0).expand(m, -1)])
            obs = env.reset()[0]["actor"]
            alive = torch.ones(num_envs, dtype=torch.bool, device=device)
            ret = torch.zeros(num_envs, device=device)
            length = torch.zeros(num_envs, device=device)
            err_xy = torch.zeros(num_envs, device=device)
            err_yaw = torch.zeros(num_envs, device=device)
            steps = 0
            for _ in range(args.horizon):
                normalizer.accumulate(obs, alive)
                x = normalizer(obs).view(P + m, K, obs_dim)
                act = net.forward(params, x).reshape(num_envs, act_dim)
                if args.action_clip is not None:
                    act = act.clamp(-args.action_clip, args.action_clip)
                obs_dict, rew, term, trunc, _ = env.step(act)
                steps += 1
                ret += rew * alive
                length += alive
                still = alive & ~(term | trunc)
                cmd = twist.vel_command_b
                err_xy += still * (cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]).norm(dim=1)
                err_yaw += still * (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
                alive = still
                obs = obs_dict["actor"]
                if not alive.any():
                    break

            fitness = ret[: P * K].view(P, K).mean(1)
            alg_stats = algo.tell(fitness)
            normalizer.update()

        torch.cuda.synchronize()
        env_steps += steps * num_envs
        alive_steps += int(length.sum().item())
        wall = time.time() - start
        c = is_center
        c_len = length[c]
        row = {
            "generation": gen,
            "wall_time_s": wall,
            "env_steps": env_steps,
            "alive_env_steps": alive_steps,
            "gen_time_s": time.time() - t0,
            "steps_this_gen": steps,
            "fitness_mean": fitness.mean().item(),
            "fitness_max": fitness.max().item(),
            "fitness_min": fitness.min().item(),
            "center_return": ret[c].mean().item(),
            "center_episode_length": c_len.mean().item(),
            "center_survival": (c_len >= steps).float().mean().item(),
            "center_err_vel_xy": (err_xy[c].sum() / c_len.sum().clamp_min(1)).item(),
            "center_err_vel_yaw": (err_yaw[c].sum() / c_len.sum().clamp_min(1)).item(),
            "theta_norm": algo.mean.norm().item(),
            **{f"algo_{k}": v for k, v in alg_stats.items()},
        }
        for k, v in row.items():
            if k != "generation":
                writer.add_scalar(f"ES/{k}", v, gen)
        writer.add_scalar("Time/center_return_vs_wall_s", row["center_return"], int(wall))
        writer.add_scalar("Time/center_return_vs_Msteps", row["center_return"],
                          int(env_steps / 1e6))
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()
        print(f"gen {gen:4d}  {wall / 60:5.1f} min  {env_steps / 1e6:7.0f}M steps  "
              f"fit {row['fitness_mean']:6.2f} (max {row['fitness_max']:6.2f})  "
              f"center {row['center_return']:6.2f} len {row['center_episode_length']:5.0f}"
              f"  err {row['center_err_vel_xy']:.2f}  [{row['gen_time_s']:.1f}s]",
              flush=True)

        done = wall > args.time_limit_min * 60 or gen == args.generations
        if gen % args.save_every == 0 or done:
            save(gen, env_steps, wall)
        if done:
            break

    writer.close()
    csv_file.close()
    env.close()
    print(f"Wrote {log_dir}")


if __name__ == "__main__":
    main()

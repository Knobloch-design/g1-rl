"""Replay saved checkpoints on one fixed course, recording video and metrics.

Every checkpoint drives the same robot (same seed, same randomization) through
the same 20 s sequence of velocity commands, with exploration noise off, so
the attempts can be compared directly.

    GALLIUM_DRIVER=d3d12 MUJOCO_GL=glfw .venv/bin/python render_progress.py logs/rsl_rl/g1_velocity/<run>

Writes to <run>/progress/:
    iter_00000.mp4 ...    one 720p video per checkpoint, with a text overlay
    metrics.csv           falls, tracking errors, distance, per checkpoint
    montage.mp4           all attempts tiled side by side (with --montage)

By default it picks 10 checkpoints spaced closer together early in training,
where behavior changes fastest. Pass --iterations 0 500 1000 ... to choose.
"""

import argparse
import csv
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
SEED = 0

# (start time s, label, [vx m/s, vy m/s, yaw rate rad/s]). All within the
# command ranges used in training before the curriculum widens them.
COURSE = [
    (0.0, "stand still", [0.0, 0.0, 0.0]),
    (3.0, "walk forward", [0.8, 0.0, 0.0]),
    (8.0, "walk + turn left", [0.5, 0.0, 0.5]),
    (12.0, "step sideways", [0.0, 0.4, 0.0]),
    (15.0, "walk backward", [-0.5, 0.0, 0.0]),
    (18.0, "stand still", [0.0, 0.0, 0.0]),
]
DURATION = 20.0  # s


def course_at(t: float) -> tuple[str, list[float]]:
    label, command = COURSE[0][1:]
    for start, phase_label, phase_command in COURSE:
        if t >= start:
            label, command = phase_label, phase_command
    return label, command


def pick_iterations(available: list[int], count: int) -> list[int]:
    """Choose `count` checkpoints, spaced geometrically (denser early)."""
    if len(available) <= count:
        return available
    last = available[-1]
    first_nonzero = next(i for i in available if i > 0)
    targets = [0] + list(np.geomspace(first_nonzero, last, count - 1))
    chosen: list[int] = []
    for target in targets:
        remaining = [i for i in available if i not in chosen]
        chosen.append(min(remaining, key=lambda i: abs(i - target)))
    return sorted(chosen)


def make_env(width: int, height: int, device: str):
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = SEED
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    # The sensor overlays (translucent foot-height rays) make each frame ~5x
    # slower to render under WSL2; keep only the command arrows.
    for sensor in env_cfg.scene.sensors or ():
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    twist = env_cfg.commands["twist"]
    # The script sets the command every step; turn off random resampling and
    # the standing/heading/world/forward modes that would override it.
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.init_velocity_prob = 0.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)


def overlay(frame: np.ndarray, lines: list[str], font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    pad, line_height = 12, font.size + 6
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle([0, 0, image.width, box_height], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_height), line, fill="white", font=font)
    return np.asarray(image)


@torch.inference_mode()
def run_attempt(venv, policy, iteration, steps_per_iter, video_path, font):
    env = venv.unwrapped
    robot = env.scene["robot"]
    twist = env.command_manager.get_term("twist")
    dt = env.step_dt
    num_steps = int(DURATION / dt)

    env.seed(SEED)
    torch.manual_seed(SEED)
    venv.reset()

    falls = 0
    first_fall = None
    lin_errors, yaw_errors, speeds = [], [], []
    writer = imageio.get_writer(
        video_path, fps=round(1 / dt), codec="libx264", quality=8,
        macro_block_size=8,
    )
    for step in range(num_steps):
        t = step * dt
        label, command = course_at(t)
        twist.vel_command_b[:] = torch.tensor(command, device=env.device)
        obs = venv.get_observations()
        _, _, dones, _ = venv.step(policy(obs))

        if dones[0]:
            falls += 1
            first_fall = t if first_fall is None else first_fall
        else:
            lin_vel = robot.data.root_link_lin_vel_b[0]
            yaw_rate = robot.data.root_link_ang_vel_b[0, 2]
            cmd = twist.vel_command_b[0]
            lin_errors.append(torch.norm(cmd[:2] - lin_vel[:2]).item())
            yaw_errors.append(abs(cmd[2] - yaw_rate).item())
            speeds.append(torch.norm(lin_vel[:2]).item())

        lines = [
            f"Iteration {iteration}  ({iteration * steps_per_iter / 1e6:.1f}M"
            f" training steps)",
            f"t = {t:4.1f} s   {label}:  vx {command[0]:+.1f}  "
            f"vy {command[1]:+.1f}  yaw {command[2]:+.1f}",
            f"falls: {falls}",
        ]
        writer.append_data(overlay(venv.unwrapped.render(), lines, font))
    writer.close()

    return {
        "iteration": iteration,
        "training_steps": iteration * steps_per_iter,
        "falls": falls,
        "time_to_first_fall_s": first_fall if first_fall is not None else "",
        "mean_lin_vel_error": float(np.mean(lin_errors)) if lin_errors else "",
        "mean_yaw_rate_error": float(np.mean(yaw_errors)) if yaw_errors else "",
        "distance_m": float(np.sum(speeds) * dt),
    }


def make_montage(videos: list[Path], out: Path, columns: int = 5) -> None:
    """Tile the videos into one grid video with ffmpeg's xstack."""
    n = len(videos)
    inputs = sum((["-i", str(v)] for v in videos), [])
    scaled = "".join(f"[{i}:v]scale=384:216[v{i}];" for i in range(n))
    layout = "|".join(
        f"{(i % columns) * 384}_{(i // columns) * 216}" for i in range(n)
    )
    stack = "".join(f"[v{i}]" for i in range(n))
    filter_graph = f"{scaled}{stack}xstack=inputs={n}:layout={layout}:fill=black"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *inputs,
         "-filter_complex", filter_graph, "-c:v", "libx264", "-crf", "23",
         str(out)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--iterations", type=int, nargs="*")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--montage", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    checkpoints = {
        int(re.fullmatch(r"model_(\d+)\.pt", p.name).group(1)): p
        for p in args.run_dir.glob("model_*.pt")
    }
    available = sorted(checkpoints)
    iterations = args.iterations or pick_iterations(available, args.count)
    missing = [i for i in iterations if i not in checkpoints]
    if missing:
        raise SystemExit(f"No checkpoint for iterations {missing}; have {available}")
    print(f"Rendering iterations: {iterations}")

    out_dir = args.run_dir / "progress"
    out_dir.mkdir(exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    venv = make_env(args.width, args.height, device)
    agent_cfg = load_rl_cfg(TASK)
    steps_per_iter = agent_cfg.num_steps_per_env * _num_train_envs(args.run_dir)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=device)
    font = ImageFont.load_default(size=max(16, args.height // 30))

    metrics_path = out_dir / "metrics.csv"
    rows = []
    if args.skip_existing and metrics_path.exists():
        with open(metrics_path) as f:
            rows = list(csv.DictReader(f))
    done = {int(r["iteration"]) for r in rows}

    videos = []
    for iteration in iterations:
        video = out_dir / f"iter_{iteration:05d}.mp4"
        videos.append(video)
        if args.skip_existing and video.exists() and iteration in done:
            print(f"iteration {iteration}: already rendered")
            continue
        runner.load(
            str(checkpoints[iteration]), load_cfg={"actor": True}, strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        result = run_attempt(venv, policy, iteration, steps_per_iter, video, font)
        rows = [r for r in rows if int(r["iteration"]) != iteration] + [result]
        print(
            f"iteration {iteration}: falls {result['falls']}, "
            f"distance {result['distance_m']:.1f} m, "
            f"lin vel error {result['mean_lin_vel_error']}"
        )

    rows.sort(key=lambda r: int(r["iteration"]))
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.montage:
        make_montage(videos, out_dir / "montage.mp4")
    venv.close()
    print(f"Wrote {out_dir}")


def _num_train_envs(run_dir: Path) -> int:
    """Read the number of parallel training envs from the saved config."""
    text = (run_dir / "params" / "env.yaml").read_text()
    match = re.search(r"^\s*num_envs:\s*(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else 1


if __name__ == "__main__":
    main()

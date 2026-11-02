"""Build an HTML report pairing training curves with the rendered attempts.

    .venv/bin/python build_report.py logs/rsl_rl/g1_velocity/<run>

Needs <run>/progress/ from render_progress.py. Exports every TensorBoard
scalar to progress/scalars.csv and writes progress/report.html, which plays
the iter_*.mp4 videos from the same folder. Optional progress/notes.json maps
an iteration to a short note shown next to its video.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (tag, title, explanation, options). Grouped into sections below.
SECTIONS = [
    ("Is it learning?", "The headline numbers. Read these first.", [
        ("Train/mean_reward", "Mean episode reward",
         "Total reward per episode, averaged over episodes that ended this iteration.", {}),
        ("Train/mean_episode_length", "Mean episode length (steps, 1000 = 20 s)",
         "How long the robot survives. 1000 means it reached the time limit without falling.", {}),
        ("Episode_Termination/fell_over", "Falls per iteration",
         "Episodes that ended in a fall, out of 4096 robots.", {}),
    ]),
    ("What is the reward made of?", "Per-episode contribution of individual reward terms.", [
        ("Episode_Reward/track_linear_velocity", "Linear velocity tracking reward",
         "The main task: move at the commanded speed.", {}),
        ("Episode_Reward/upright", "Upright reward",
         "Keep the torso vertical.", {}),
        ("Episode_Reward/action_rate_l2", "Action-rate penalty",
         "Penalty for jerky changes in joint targets; grows as the robot does more.", {}),
    ]),
    ("Exploration", "How random the policy's actions are.", [
        ("Policy/mean_std", "Action noise std",
         "Std of the Gaussian added to actions (in units of 0.25 rad per joint). Shrinks as the policy commits.", {}),
        ("Loss/entropy", "Policy entropy",
         "Same idea as the std, as entropy. The 0.01 entropy bonus slows its decline.", {}),
    ]),
    ("The critic", "The value network that predicts future reward, used to score actions.", [
        ("Loss/value", "Value loss",
         "Squared error of the critic's predictions. Not expected to go to zero: the targets move as the policy improves.", {"log": True}),
        ("Diagnostics/explained_variance", "Explained variance",
         "Share of return variance the critic explains. 1 is perfect, 0 is no better than the mean.", {"ymin": 0, "ymax": 1}),
        ("Diagnostics/value_mean", "Mean predicted value",
         "Average critic prediction. Rises as the robot earns more reward.", {}),
    ]),
    ("Update health", "Whether each PPO update was a safe-sized step.", [
        ("Diagnostics/approx_kl", "KL divergence per update",
         "How far one update moved the policy. The adaptive learning rate aims at 0.01.", {"ref": 0.01}),
        ("Diagnostics/clip_fraction", "Clip fraction",
         "Share of samples where PPO's clipping kicked in. 0.1 to 0.3 is typical.", {}),
        ("Loss/learning_rate", "Learning rate",
         "Adjusted every mini-batch: halved when KL is too high, raised when too low.", {"log": True}),
        ("Diagnostics/grad_norm_actor", "Actor gradient norm (before clipping)",
         "Size of the actor's gradient. Values above 1.0 get clipped to 1.0.", {"ref": 1.0}),
        ("Diagnostics/grad_clipped_frac", "Share of gradients clipped",
         "Fraction of mini-batch gradients (actor and critic) that hit the 1.0 limit.", {"ymin": 0, "ymax": 1}),
    ]),
]

STRIDE = 2  # keep every 2nd iteration in the page


def load_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    data: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for path in sorted(run_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for event in acc.Scalars(tag):
                data[tag][event.step] = event.value
    return data


def write_csv(data, path: Path) -> None:
    tags = sorted(data)
    steps = sorted(set().union(*(d.keys() for d in data.values())))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", *tags])
        for step in steps:
            writer.writerow([step, *(data[t].get(step, "") for t in tags)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.run_dir / "progress"

    data = load_scalars(args.run_dir)
    write_csv(data, out_dir / "scalars.csv")

    sections = []
    series = {}
    for title, blurb, charts in SECTIONS:
        kept = []
        for tag, chart_title, explain, opts in charts:
            if tag not in data:
                continue
            points = sorted(data[tag].items())
            last = points[-1][0]
            series[tag] = [
                [step, value] for step, value in points
                if step % STRIDE == 0 or step == last
            ]
            kept.append({"tag": tag, "title": chart_title, "explain": explain, **opts})
        sections.append({"title": title, "blurb": blurb, "charts": kept})

    with open(out_dir / "metrics.csv") as f:
        attempts = list(csv.DictReader(f))
    notes_path = out_dir / "notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    for attempt in attempts:
        attempt["note"] = notes.get(attempt["iteration"], "")
        still = out_dir / f"still_{int(attempt['iteration']):05d}.jpg"
        attempt["still"] = still.name if still.exists() else ""

    payload = json.dumps(
        {"sections": sections, "series": series, "attempts": attempts,
         "run": args.run_dir.name, "intro": notes.get("intro", "")},
        separators=(",", ":"),
    )
    template = (Path(__file__).parent / "report_template.html").read_text()
    (out_dir / "report.html").write_text(template.replace("__DATA__", payload))
    print(f"Wrote {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()

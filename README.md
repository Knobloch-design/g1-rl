# g1-rl

Learning to walk with the Unitree G1 humanoid using reinforcement learning,
set up as a learning experiment: every part of training is logged, and saved
checkpoints are replayed on a fixed course to watch the policy improve.

| Folder | What it is |
|---|---|
| [`mjlab/`](mjlab) | GPU training with [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) and rsl_rl PPO, plus extra PPO diagnostics, progress videos and an HTML report. Start with [`mjlab/SETUP.md`](mjlab/SETUP.md). |
| [`drake-env/`](drake-env) | A Gymnasium environment for the same G1 in [Drake](https://drake.mit.edu) with the CENIC integrator, for slower but more accurate sim-to-sim checks. |

## First run (2026-10-31)

3000 PPO iterations (295M steps, 4096 parallel robots, ~2.5 h on a
GTX 1660 Super). On the 20 s evaluation course the final policy walks with
no falls, 0.10 m/s speed-tracking error and 0.06 rad/s turn-rate error.

Results are in
[`mjlab/logs/rsl_rl/g1_velocity/2026-10-31_18-05-16_tracked/`](mjlab/logs/rsl_rl/g1_velocity/2026-10-31_18-05-16_tracked):
final checkpoint (`model_2999.pt`, plus ONNX), TensorBoard event files, the
exact configs in `params/`, and `progress/` with `montage.mp4`, `metrics.csv`,
`scalars.csv` and `report.html`. The ten full-length attempt videos the report
plays are not committed; regenerate them with `render_progress.py` (see
SETUP.md).

Training went through four stages: flailing (iterations 0–25), falling on
purpose to end episodes early (25–100), standing still and ignoring commands
(150–550), then a sudden switch to walking (520–620), followed by polish.

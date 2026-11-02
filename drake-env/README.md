# drake-env (`g1_walking`)

Reinforcement learning for the Unitree G1 humanoid, simulated in
[Drake](https://drake.mit.edu) with the CENIC error-controlled integrator.
The robot model and simulator setup are copied from the
cenic-bench project (see
[`g1_walking/models/unitree_g1/README.md`](g1_walking/models/unitree_g1/README.md)).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[train]" pytest ruff
```

## Usage

```bash
.venv/bin/python -m pytest                       # sanity tests
.venv/bin/python examples/play.py                # view the G1 in meshcat
.venv/bin/python examples/train_ppo.py --num-envs 10
.venv/bin/tensorboard --logdir runs
.venv/bin/python examples/play.py --model runs/ppo_g1/final.zip \
    --vecnormalize runs/ppo_g1/vecnormalize.pkl --command 0.5 0 0
```

## Environment

`G1WalkEnv` (registered as `G1Walk-v0`, 1000-step / 20 s limit) is a standard
Gymnasium env in [`g1_walking/env.py`](g1_walking/env.py):

- **Task:** track a random `[vx, vy, yaw_rate]` command sampled each episode.
- **Action (29):** joint position offsets from mjlab's knees-bent home pose,
  times `action_scale` (0.25 rad), sent to the PD controllers in `g1.sdf` at
  50 Hz.
- **Observation (99):** pelvis linear and angular velocity and gravity in the
  pelvis frame, the command, joint positions relative to home, joint
  velocities, and the last action.
- **Reward:** legged_gym-style velocity tracking plus penalties; weights are
  in `G1WalkEnv.reward_weights`, and each term is in `info["reward_terms"]`.
- **Termination:** pelvis below 0.45 m, tilt above 1 rad, or a simulator
  failure.

The plant is continuous-time (`time_step=0.0`) as CENIC requires, and the
ground is a rigid hydroelastic half-space.

## Speed

On one core with the default `accuracy=1e-3`: about 127 steps/s when holding
the home pose and about 33 steps/s with random actions (CENIC takes small
steps through chaotic contact). `accuracy=1e-2`, the training default, gives
about 197 and 94. Run envs in parallel with `--num-envs`.

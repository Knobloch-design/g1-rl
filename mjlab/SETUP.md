# G1 walking with mjlab — how this experiment is set up

This folder trains a walking policy for the Unitree G1 humanoid with
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo physics running on the
GPU via MuJoCo Warp) and PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
It is set up as a learning experiment: every part of training is logged, and
saved checkpoints are replayed on a fixed course so you can watch the policy
improve.

```
mjlab/
  .venv/               Python 3.13 environment (mjlab 1.6.0, torch CUDA)
  train_tracked.py     mjlab's `train` + extra PPO diagnostics (gradients, KL…)
  render_progress.py   replays checkpoints on a fixed course → videos + metrics
  build_report.py      curves + attempts → progress/report.html
  report_template.html page template used by build_report.py
  SETUP.md             this file
  logs/rsl_rl/g1_velocity/<date>_<run>/
      events.out.tfevents.*   TensorBoard logs
      model_<iter>.pt         checkpoints, every 50 iterations
      params/env.yaml         the exact environment config used
      params/agent.yaml       the exact PPO config used
      progress/               videos + metrics.csv from render_progress.py,
                              report.html from build_report.py
```

## Recreating the environment

mjlab needs Python 3.10–3.13 and is best installed with
[uv](https://docs.astral.sh/uv/) (plain pip on newer Pythons picks up an
ancient release). From this folder:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv mjlab==1.6.0 tensorboard imageio imageio-ffmpeg pillow
```

Tested with torch 2.15.0, mujoco / mujoco-warp 3.11.0, warp-lang 1.18.0 and
rsl-rl-lib 5.4.2.

Everything below describes mjlab's built-in task `Mjlab-Velocity-Flat-Unitree-G1`,
unchanged. The only things added here are logging and evaluation.

---

## 1. The problem the robot is solving

**Goal:** follow a velocity command — "walk forward at 0.8 m/s", "turn left at
0.5 rad/s", "stand still" — without falling.

This is framed as a reinforcement learning problem (a Markov decision process):
every 0.02 s (50 Hz) the policy sees an **observation**, outputs an **action**,
the simulator advances, and the robot receives a **reward**. PPO adjusts the
policy to collect more reward over time.

### Simulation

| | |
|---|---|
| Physics | MuJoCo Warp on the GPU, timestep 0.005 s (200 Hz) |
| Control | policy runs every 4 physics steps → 50 Hz |
| Parallel robots | 4,096 independent copies simulated at once |
| Terrain | flat plane |
| Episode length | 20 s (1,000 policy steps), or until the robot falls |
| Robot | G1, 29 actuated joints, starting in a knees-bent stance |

### Observations — what the policy sees (99 numbers)

| Term | Size | Meaning |
|---|---|---|
| `base_lin_vel` | 3 | pelvis velocity (IMU), noise ±0.5 |
| `base_ang_vel` | 3 | pelvis rotation rate (IMU), noise ±0.2 |
| `projected_gravity` | 3 | which way is "down" in the pelvis frame → tells it how tilted it is |
| `joint_pos` | 29 | joint angles relative to the default stance, noise ±0.01 |
| `joint_vel` | 29 | joint speeds, noise ±1.5 |
| `actions` | 29 | the previous action (gives the policy short-term memory) |
| `command` | 3 | the requested [forward, sideways, turn] velocity |

The noise is deliberate: real sensors are noisy, and training with noise
produces a policy that doesn't rely on perfect information.

**The critic sees more (111 numbers).** PPO trains two networks: the *actor*
(the policy) and the *critic* (which estimates how good a situation is). The
critic is only used during training, so it's allowed "privileged" information
the real robot wouldn't have: noise-free joint angles, plus foot height, foot
air time, foot contact and foot contact forces. This is called an
*asymmetric actor-critic* and makes the critic's job easier.

### Actions — what the policy controls (29 numbers)

One number per joint, turned into a **target joint angle**:

```
target = default_stance_angle + scale × action
```

Each joint's built-in PD controller then produces the torque to reach that
target. The scale is per joint (`0.25 × effort_limit / stiffness`), so an
action of 1 asks each joint for roughly a quarter of what its motor can push
against its stiffness. Position targets are much easier to learn than raw
torques.

### Commands — what the robot is asked to do

Each robot gets a new random command every 3–8 seconds:

- forward speed in [-1, 1] m/s, sideways speed in [-1, 1] m/s, turn rate in [-0.5, 0.5] rad/s
- 10% of robots are told to stand still
- 30% are given a heading to face (the turn rate is computed to steer there)
- 20% are told to walk straight forward only

A **curriculum** widens the speed range later (to [-1.5, 2.0] m/s at iteration
5,000 and [-2, 3] m/s at 10,000). A 3,000-iteration run never reaches those
stages.

### Rewards — what counts as "good" (summed each step)

| Term | Weight | Encourages / penalizes |
|---|---|---|
| `track_linear_velocity` | +2.0 | moving at the commanded forward/sideways speed |
| `track_angular_velocity` | +2.0 | turning at the commanded rate |
| `upright` | +1.0 | keeping the torso vertical |
| `pose` | +1.0 | staying near the default posture (looser while walking/running) |
| `foot_clearance` | −2.0 | feet not lifting ~10 cm when stepping |
| `foot_swing_height` | −0.25 | wrong peak swing height |
| `foot_slip` | −0.1 | feet sliding while on the ground |
| `soft_landing` | −0.00001 | hard foot impacts |
| `dof_pos_limits` | −1.0 | joints hitting their limits |
| `action_rate_l2` | −0.1 | jerky actions (big change between steps) |
| `body_ang_vel` | −0.05 | torso wobbling |
| `angular_momentum` | −0.02 | whole-body spinning |
| `self_collisions` | −1.0 | limbs hitting each other |

The positive terms say *what* to do; the penalties shape *how* — smooth,
upright, with clean steps. Each term is logged separately (`Episode_Reward/*`),
so you can watch, e.g., tracking improve before foot clearance does.

### When an episode ends

- **fell over:** torso tilted more than 70° → episode ends (this is the main
  failure signal early in training)
- **time out:** 20 s passed → episode ends normally

### Domain randomization — making the policy robust

- foot friction randomized between 0.3 and 1.2
- small joint encoder bias (±0.015 rad)
- torso center of mass shifted by up to ±2.5 cm
- random **pushes** every 1–3 s (sudden velocity changes)
- random start position and heading

---

## 2. The model and the learning algorithm

### Networks

| | Actor (policy) | Critic (value function) |
|---|---|---|
| Input | 99 observations | 111 observations |
| Hidden layers | 512 → 256 → 128, ELU | 512 → 256 → 128, ELU |
| Output | 29 action means + one shared noise level (std) | 1 number: expected future reward |
| Input normalization | running mean/std | running mean/std |

During training, actions are sampled from a Gaussian around the actor's
output; the noise level (`std`, starts at 1.0) is itself learned and shrinks as
the policy becomes confident. Evaluation videos use the mean action (no noise).

### PPO, one iteration at a time

1. **Collect:** all 4,096 robots run 24 steps → **98,304 samples** per iteration.
2. **Score:** compute each sample's *advantage* — how much better it turned out
   than the critic predicted (GAE, γ = 0.99, λ = 0.95).
3. **Update:** 5 passes (epochs) over the data in 4 mini-batches = 20 gradient
   steps. PPO's *clipping* (ε = 0.2) stops any single update from changing the
   policy too much.
4. **Adapt the learning rate:** if the policy changed more than the target
   (KL divergence > 0.02) the learning rate shrinks; if it barely changed
   (< 0.005) it grows. Starts at 1e-3.

Other settings: value loss coefficient 1.0 (clipped), entropy bonus 0.01,
gradient norm clipped at 1.0.

**Run length:** 3,000 iterations ≈ 295 million samples ≈ 3 hours on this GPU
(GTX 1660 Super, ~27,000 samples/s).

---

## 3. What gets logged, and how to read it

Open TensorBoard while training runs (or afterwards):

```bash
.venv/bin/tensorboard --logdir logs
```

then open <http://localhost:6006>.

### Progress — is it learning?

| Graph | What it means | What to expect |
|---|---|---|
| `Train/mean_reward` | average total reward per episode | rises, then plateaus |
| `Train/mean_episode_length` | how long robots survive (max 1,000 steps) | climbs from ~50 to near 1,000 as falls stop |
| `Episode_Termination/fell_over` | how many episodes end in a fall | drops toward 0 |
| `Metrics/twist/error_vel_xy` | m/s gap between commanded and actual velocity | falls as it learns to walk on command |
| `Metrics/twist/error_vel_yaw` | same for turning | falls |
| `Episode_Reward/*` | each reward term separately | survival terms improve first, gait-quality terms later |

### Losses — how the update is going

| Graph | What it means | Healthy pattern |
|---|---|---|
| `Loss/value` | how wrong the critic's predictions are | spikes early (returns change fast), then settles |
| `Loss/surrogate` | PPO's policy objective | small and noisy, around 0 — not meant to steadily decrease |
| `Loss/entropy` | how random the policy is | decreases slowly as it commits to a gait |
| `Loss/learning_rate` | the adaptive learning rate | moves up/down to keep KL near 0.01 |
| `Policy/mean_std` | action noise level | shrinks from 1.0 |

Unlike supervised learning, RL losses don't simply go down: the data changes as
the policy changes. The *progress* graphs are the real measure of success.

### Diagnostics — added by `train_tracked.py`

| Graph | What it means | Healthy pattern |
|---|---|---|
| `Diagnostics/grad_norm_actor`, `grad_norm_critic` | size of the gradient before clipping | stable; persistent spikes mean unstable training |
| `Diagnostics/grad_clipped_frac` | fraction of steps where the gradient hit the 1.0 cap | low-ish; near 1.0 all the time means steps are always being cut short |
| `Diagnostics/approx_kl` | how much one update changed the policy | near the 0.01 target |
| `Diagnostics/clip_fraction` | share of samples where PPO's clipping kicked in | roughly 0.1–0.3 |
| `Diagnostics/explained_variance` | how well the critic predicts returns (1 = perfect) | rises from ~0 toward 0.8–1.0 |
| `Diagnostics/advantage_std` | spread of "better/worse than expected" | — |
| `Diagnostics/return_mean`, `value_mean` | target vs. predicted value | should track each other |

### Histograms (every 50 iterations)

`Weights/actor/...`, `Weights/critic/...` and `Gradients/...` show the
distribution of every layer's weights and gradients over time (TensorBoard's
*Histograms* and *Distributions* tabs). Look for weights drifting smoothly,
and for gradients that neither vanish to 0 nor explode.
`Weights/actor/distribution/std_param` is the learned action noise.

### Performance

`Perf/total_fps` (samples per second), `Perf/collection_time` (simulation),
`Perf/learning_time` (network update).

---

## 4. Watching attempts: the fixed evaluation course

Checkpoints are saved every 50 iterations (61 over a 3,000-iteration run).
After training, `render_progress.py` replays 10 of them — chosen closer
together early on, when behavior changes fastest — on the **same 20 s course**,
with the same random seed and no action noise:

| Time | Command |
|---|---|
| 0–3 s | stand still |
| 3–8 s | walk forward 0.8 m/s |
| 8–12 s | walk 0.5 m/s while turning left 0.5 rad/s |
| 12–15 s | step sideways 0.4 m/s |
| 15–18 s | walk backward 0.5 m/s |
| 18–20 s | stand still |

If the robot falls it respawns and the course continues; falls are counted.

```bash
GALLIUM_DRIVER=d3d12 MUJOCO_GL=glfw .venv/bin/python render_progress.py logs/rsl_rl/g1_velocity/<run> --montage
.venv/bin/python build_report.py logs/rsl_rl/g1_velocity/<run>   # → progress/report.html
```

Output in `<run>/progress/`:

- `iter_XXXXX.mp4` — 720p video per checkpoint, overlaid with the iteration,
  current command and fall count. The green arrow is the commanded velocity.
- `metrics.csv` — falls, time to first fall, mean velocity/turn tracking error
  and distance walked, per checkpoint.
- `montage.mp4` — all 10 attempts tiled side by side.

Pick specific checkpoints with `--iterations 0 100 250 ...`. Rendering takes
about 2.5 minutes per checkpoint.

---

## 5. Commands

Train (3,000 iterations, tracked):

```bash
.venv/bin/python train_tracked.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --agent.run-name tracked
```

Continue a run for longer (from its last checkpoint):

```bash
.venv/bin/python train_tracked.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.max-iterations 2000 --agent.logger tensorboard --agent.resume True --agent.load-run <run folder name>
```

Watch a checkpoint interactively (opens a window):

```bash
GALLIUM_DRIVER=d3d12 .venv/bin/play Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt
```

---

## 6. Environment notes (WSL2 + GTX 1660 Super)

- **Python 3.13 via uv.** mjlab needs Python < 3.14; the system Python is 3.14,
  where `pip install mjlab` silently installs an ancient 0.0.4 release.
  `uv` (in `~/.local/share/uv-venv`, linked from `~/.local/bin`) created this
  venv with its own Python 3.13.
- **`MUJOCO_GL=glfw`** — MuJoCo's default headless EGL rendering can't access
  the GPU under WSL2; GLFW renders through WSLg's display instead.
- **`GALLIUM_DRIVER=d3d12`** — without it, OpenGL falls back to `llvmpipe`
  (software rendering on the CPU). With it, rendering runs on the GTX 1660
  through DirectX.
- **Sensor debug overlays are off in evaluation videos** — the translucent
  foot-height rays made rendering ~5× slower under WSL2.
- **4,096 parallel envs** use ~1.6 GB of the 6 GB GPU (the Windows display uses
  ~2.7 GB). 8,192 is only 13% faster and nearly fills it.
- Keep Windows from sleeping during a run, and avoid heavy GPU use (games,
  video) alongside it.

## 7. Relation to the Drake environment

`../drake-env` has a Gymnasium environment for the same G1 in Drake with the
CENIC integrator — accurate, but about 15–30× slower. The plan is to train
here and use Drake to check that the learned walking isn't an artifact of one
simulator (a "sim-to-sim" check). Its observation layout mirrors mjlab's, but
joint order and scaling must be matched before a policy can be transferred.

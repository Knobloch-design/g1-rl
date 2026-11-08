# PPO vs evolutionary methods on the G1 walking task

Goal: train the same G1 velocity-tracking task with gradient-free
(evolutionary) optimizers and with PPO using different gradient optimizers,
and compare **how fast** each learns (wall-clock time and simulated steps)
and **how good** the resulting walking is.

## Files

| File | What it does |
|---|---|
| `train_es.py` | ES training loop (one generation = propose candidates, run every candidate's episodes in parallel on the GPU, update) |
| `es/algorithms.py` | ARS, OpenAI-ES, PGPE, CMA-ES, sep-CMA-ES; Adam / SGD / ClipUp update rules |
| `es/policy.py` | flat-vector MLP / linear policies evaluated for a whole population in one batched matmul; observation normalizer |
| `es/env.py` | the training env (curriculum off, see below) |
| `eval_runs.py` | replays checkpoints of any PPO or ES run on identical seeded tests → `<run>/eval.csv` |
| `compare_runs.py` | all `eval.csv` files → `logs/comparison/index.html` |
| `run_overnight.sh` | the 16-hour experiment queue with a deadline guard |

## The methods

**Gradient-based (PPO).** Backpropagates through an actor and a critic network.
The *optimizer* is the rule that turns a gradient into a parameter step:

- **Adam** (rsl_rl default): per-parameter step sizes from running gradient
  statistics; lr 1e-3 with the adaptive-KL schedule.
- **SGD**: step = lr × gradient. Run twice: with the default adaptive schedule
  (which caps lr at 1e-2, so SGD's steps are far smaller than Adam's) and
  with a fixed, larger lr. Both are clipped to gradient norm 1.0.

**Evolutionary / black-box.** No backprop and no critic: perturb the policy's
parameters, run episodes, and move toward perturbations that earned more
return. Each sees only one number per candidate (its episode return).

| Method | Update | Learns its own noise? |
|---|---|---|
| ARS | step along reward differences of the best antithetic directions, scaled by reward std (SGD-like) | no |
| OpenAI-ES | rank-shaped gradient estimate → Adam (or SGD), weight decay | no |
| PGPE | rank-shaped gradient estimate → ClipUp (or Adam) | per-parameter σ |
| CMA-ES | weighted recombination of the best candidates | full covariance + step size |
| sep-CMA-ES | same, diagonal covariance only (scales to big nets) | diagonal + step size |

## Fairness decisions

- **Same task for everyone.** mjlab's command-speed curriculum advances by env
  steps. ES burns env steps much faster per update than PPO, so it would reach
  the harder speed ranges early; it is turned off for ES. PPO runs stop at
  4000 iterations, before PPO's curriculum would kick in (5000).
- **Same evaluation for everyone.** Training curves aren't comparable (PPO logs
  its noisy policy's return, ES its candidates' return on a 10 s horizon), so
  `eval_runs.py` replays checkpoints deterministically with fixed seeds on
  (a) 1024 random-command full 20 s episodes and (b) the fixed 20 s course.
- **Network size.** Runs at three sizes: PPO-sized MLP 512-256-128 (219k
  parameters), small MLP 32-32 (5.2k), linear (2.9k). Full CMA-ES only runs on
  the small MLP: its covariance matrix for the big one would have ~48 billion
  entries. PPO "small" shrinks only the actor (the policy); its critic is
  training-only and stays big.
- **ES horizon 500 steps (10 s)** during training to get twice as many
  generations; evaluation always uses full 20 s episodes.
- **One seed per configuration.** Fine for learning about the methods; real
  claims would need 3–5 seeds.

## Settings used overnight (2026-11-08)

See `run_overnight.sh` for the exact commands. ES runs: 55 min each;
antithetic methods 2048 candidates × 2 episodes; CMA-ES 256 candidates × 16
episodes; 256 extra robots run the mean policy each generation.
PPO runs: 4096 robots, 4000 iterations (~25 min).

## Reading the results

- `logs/queue/status.txt`: which runs started, finished or were skipped
- `logs/queue/<name>.log`: each run's console output
- `logs/comparison/index.html`: the comparison page (open in a browser)
- `tensorboard --logdir logs`: every run's training curves (`ES/*` for ES runs)
- videos of any checkpoint: `render_progress.py` works for PPO runs; for ES
  runs, ask Claude to extend it (ES checkpoints load with `es.policy.ESPolicy`)

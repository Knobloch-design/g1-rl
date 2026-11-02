"""mjlab's `train` command with extra PPO diagnostics logged to TensorBoard.

Usage is identical to `train`, e.g.

    MUJOCO_GL=glfw .venv/bin/python train_tracked.py \
        Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096 \
        --agent.max-iterations 3000 --agent.logger tensorboard

On top of what rsl_rl already logs (Loss/*, Policy/*, Train/*, Episode_*,
Metrics/*, Perf/*), this adds, every iteration:

    Diagnostics/grad_norm_actor        L2 norm of the actor gradient before
    Diagnostics/grad_norm_critic       clipping, averaged over mini-batches
    Diagnostics/grad_norm_actor_max    largest actor gradient norm this update
    Diagnostics/grad_clipped_frac      fraction of mini-batches whose gradient
                                       exceeded max_grad_norm and was clipped
    Diagnostics/approx_kl              KL(old policy || updated policy)
    Diagnostics/clip_fraction          fraction of samples whose probability
                                       ratio left PPO's [1-eps, 1+eps] band
    Diagnostics/explained_variance     how much of the return variance the
                                       critic explains (1 = perfect, <=0 = none)
    Diagnostics/advantage_std          spread of raw advantages
    Diagnostics/return_mean            mean bootstrapped return target
    Diagnostics/value_mean             mean critic prediction

and every `HISTOGRAM_EVERY` iterations, weight and gradient histograms of
every actor and critic layer (including the learned action noise,
`actor/distribution/std_param`).

Nothing in mjlab or rsl_rl is modified; PPO.update and Logger.log are wrapped
at import time.
"""

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter

HISTOGRAM_EVERY = 50  # iterations; matches the checkpoint interval
DIAG_SAMPLES = 16384  # rollout samples used for KL / clip fraction

_diag: dict[str, float] = {}
_state: dict = {"alg": None, "writer": None}

_original_update = PPO.update
_original_log = Logger.log
_original_clip = torch.nn.utils.clip_grad_norm_


def _tracked_update(self: PPO) -> dict[str, float]:
    st = self.storage
    with torch.no_grad():
        values = st.values.flatten()
        returns = st.returns.flatten()
        advantages_raw = (st.returns - st.values).flatten()
        n = values.numel()
        idx = torch.randperm(n, device=values.device)[:DIAG_SAMPLES]
        obs = st.observations.flatten(0, 1)[idx].clone()
        actions = st.actions.flatten(0, 1)[idx].clone()
        old_log_prob = st.actions_log_prob.flatten(0, 1)[idx].squeeze(-1)
        old_params = tuple(
            p.flatten(0, 1)[idx].clone() for p in st.distribution_params
        )

    # rsl_rl clips the actor and critic gradients separately with
    # clip_grad_norm_, which returns the norm *before* clipping.
    actor_ids = {id(p) for p in self.actor.parameters()}
    norms: dict[str, list[torch.Tensor]] = {"actor": [], "critic": []}

    def recording_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        total = _original_clip(parameters, max_norm, *args, **kwargs)
        key = "actor" if id(parameters[0]) in actor_ids else "critic"
        norms[key].append(total.detach())
        return total

    torch.nn.utils.clip_grad_norm_ = recording_clip
    try:
        loss_dict = _original_update(self)
    finally:
        torch.nn.utils.clip_grad_norm_ = _original_clip

    with torch.no_grad():
        self.actor(obs, stochastic_output=True)
        new_log_prob = self.actor.get_output_log_prob(actions).flatten()
        new_params = self.actor.output_distribution_params
        kl = self.actor.get_kl_divergence(old_params, new_params).mean()
        ratio = torch.exp(new_log_prob - old_log_prob)
        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
        explained_variance = 1.0 - (returns - values).var() / (
            returns.var() + 1e-8
        )
        actor_norms = torch.stack(norms["actor"])
        critic_norms = torch.stack(norms["critic"])

    _diag.clear()
    _diag.update(
        {
            "grad_norm_actor": actor_norms.mean().item(),
            "grad_norm_actor_max": actor_norms.max().item(),
            "grad_norm_critic": critic_norms.mean().item(),
            "grad_clipped_frac": (
                torch.cat([actor_norms, critic_norms]) > self.max_grad_norm
            )
            .float()
            .mean()
            .item(),
            "approx_kl": kl.item(),
            "clip_fraction": clip_fraction.item(),
            "explained_variance": explained_variance.item(),
            "advantage_std": advantages_raw.std().item(),
            "return_mean": returns.mean().item(),
            "value_mean": values.mean().item(),
        }
    )
    _state["alg"] = self
    return loss_dict


def _tracked_log(self: Logger, *args, **kwargs) -> None:
    _original_log(self, *args, **kwargs)
    if self.writer is None or not _diag:
        return
    it = kwargs["it"] if "it" in kwargs else args[0]
    if _state["writer"] is None:
        # A second event file in the same run directory; TensorBoard merges
        # it with rsl_rl's into one run.
        _state["writer"] = SummaryWriter(log_dir=self.log_dir)
    writer = _state["writer"]
    for key, value in _diag.items():
        writer.add_scalar(f"Diagnostics/{key}", value, it)

    alg = _state["alg"]
    if alg is not None and it % HISTOGRAM_EVERY == 0:
        for net_name, net in (("actor", alg.actor), ("critic", alg.critic)):
            for name, param in net.named_parameters():
                tag = name.replace(".", "/")
                writer.add_histogram(f"Weights/{net_name}/{tag}", param, it)
                if param.grad is not None:
                    writer.add_histogram(
                        f"Gradients/{net_name}/{tag}", param.grad, it
                    )
    writer.flush()


PPO.update = _tracked_update
Logger.log = _tracked_log


if __name__ == "__main__":
    from mjlab.scripts.train import main

    main()

"""Policies stored as flat parameter vectors, evaluated for a whole population.

Evolutionary methods treat the policy as one long vector of numbers theta and
only ever ask "how much reward does theta earn?". To evaluate thousands of
candidate vectors at once, every candidate gets its own group of simulated
robots, and one batched matrix multiply (torch.baddbmm) runs every candidate's
network on its own robots in a single GPU call.

Architecture: obs (99) -> normalize -> [hidden layers, ELU] -> actions (29).
`hidden=()` gives a linear policy (actions = W @ obs + b), as in ARS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ObsNormalizer:
    """Running mean / std of observations (shared by the whole population).

    Statistics are frozen while a generation is evaluated and updated between
    generations, so every candidate in a generation sees the same normalization
    (as in ARS-V2 and OpenAI-ES).
    """

    def __init__(self, dim: int, device, clip: float = 5.0, eps: float = 1e-2):
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.var = torch.ones(dim, device=device, dtype=torch.float64)
        self.count = 0.0
        self.clip = clip
        self.eps = eps
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._sum = torch.zeros_like(self.mean)
        self._sumsq = torch.zeros_like(self.mean)
        self._n = torch.zeros((), device=self.mean.device, dtype=torch.float64)

    def accumulate(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        """Collect statistics of `obs` [N, dim] (rows where `mask` is True)."""
        x = obs.double()
        if mask is not None:
            x = x[mask]
        self._sum += x.sum(0)
        self._sumsq += (x * x).sum(0)
        self._n += x.shape[0]

    def update(self):
        """Fold the collected statistics into the running mean / variance."""
        n = self._n.item()
        if n > 0:
            batch_mean = self._sum / n
            batch_var = (self._sumsq / n - batch_mean**2).clamp_min(0)
            total = self.count + n
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (n / total)
            self.var = (
                self.var * self.count + batch_var * n + delta**2 * self.count * n / total
            ) / total
            self.count = total
        self._reset_accumulators()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        std = (self.var.sqrt() + self.eps).float()
        return ((obs - self.mean.float()) / std).clamp(-self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].to(self.mean.device, torch.float64)
        self.var = state["var"].to(self.var.device, torch.float64)
        self.count = float(state["count"])


class FlatMLP:
    """Describes how a flat parameter vector maps onto MLP layers."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, tuple(hidden)
        sizes = [obs_dim, *self.hidden, act_dim]
        self.layers = list(zip(sizes[:-1], sizes[1:]))
        self.num_params = sum(i * o + o for i, o in self.layers)

    def init_params(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Initial parameter vector.

        Linear policy: all zeros (ARS starts here: the robot just holds its
        default stance). MLP: scaled-uniform hidden layers like PyTorch's
        default, and a near-zero output layer so initial actions are tiny.
        """
        if not self.hidden:
            return torch.zeros(self.num_params)
        parts = []
        for k, (i, o) in enumerate(self.layers):
            bound = 1.0 / math.sqrt(i)
            w = (torch.rand(i, o, generator=generator) * 2 - 1) * bound
            if k == len(self.layers) - 1:
                w *= 0.01
            parts += [w.flatten(), torch.zeros(o)]
        return torch.cat(parts)

    def unflatten(self, theta: torch.Tensor):
        """theta [P, D] -> list of (W [P, in, out], b [P, out])."""
        out, k = [], 0
        for i, o in self.layers:
            w = theta[:, k : k + i * o].view(-1, i, o)
            k += i * o
            b = theta[:, k : k + o]
            k += o
            out.append((w, b))
        return out

    def forward(self, theta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Run P networks on their own robots.

        theta: [P, D] parameters, obs: [P, K, obs_dim] (K robots per network).
        Returns actions [P, K, act_dim].
        """
        x = obs
        layers = self.unflatten(theta)
        for n, (w, b) in enumerate(layers):
            x = torch.baddbmm(b.unsqueeze(1), x, w)
            if n < len(layers) - 1:
                x = F.elu(x)
        return x


class ESPolicy(torch.nn.Module):
    """A single trained policy (the search distribution's mean), for evaluation.

    Call it with mjlab's observation TensorDict (like rsl_rl's inference
    policy) or with the raw actor observation tensor [N, obs_dim].
    """

    def __init__(self, checkpoint: dict, device):
        super().__init__()
        cfg = checkpoint["config"]
        self.net = FlatMLP(cfg["obs_dim"], cfg["act_dim"], tuple(cfg["hidden"]))
        self.theta = checkpoint["theta"].to(device).float().unsqueeze(0)
        self.normalizer = ObsNormalizer(cfg["obs_dim"], device)
        self.normalizer.load_state_dict(checkpoint["obs_norm"])
        self.action_clip = cfg.get("action_clip")

    @torch.no_grad()
    def forward(self, obs) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = obs["actor"]
        x = self.normalizer(obs).unsqueeze(0)
        a = self.net.forward(self.theta, x)[0]
        if self.action_clip is not None:
            a = a.clamp(-self.action_clip, self.action_clip)
        return a

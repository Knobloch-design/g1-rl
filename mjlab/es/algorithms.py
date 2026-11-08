"""Black-box optimizers over a flat policy parameter vector theta.

All four share one loop, driven by train_es.py:

    candidates = algo.ask()            # [popsize, D] parameter vectors to try
    fitness    = evaluate(candidates)  # [popsize] episode return of each
    stats      = algo.tell(fitness)    # move the search distribution
    algo.mean                          # current best guess (the "center")

None of them use gradients of the network or a value function: they only see
one number (the return) per candidate. They differ in how they turn those
numbers into an update.

    ARS        Augmented Random Search (Mania et al. 2018). Antithetic random
               directions, keep the best b, step along their reward
               differences scaled by the reward std. Plain SGD, fixed noise.
    OpenAI-ES  Salimans et al. 2017. Antithetic Gaussian noise, centered-rank
               fitness shaping, gradient estimate fed to Adam, weight decay.
    PGPE       Sehnke et al. 2010 with ClipUp (Toklu et al. 2020). Like
               OpenAI-ES but also *learns a per-parameter noise scale*.
    CMA-ES     Hansen 2016. Learns a full covariance matrix of the search
               distribution plus a global step size. Cost grows with D^2
               (memory) and D^3 (eigendecomposition), so `diagonal=True`
               (sep-CMA-ES, Ros & Hansen 2008) is offered for large networks.
"""

from __future__ import annotations

import math

import torch


def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    """Replace values by their rank, scaled to [-0.5, 0.5].

    Makes the update invariant to the scale of rewards and robust to outliers
    (one lucky episode can't dominate the gradient estimate).
    """
    ranks = torch.empty_like(x)
    ranks[x.argsort()] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    return ranks / (len(x) - 1) - 0.5


class Adam:
    def __init__(self, dim, device, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = torch.zeros(dim, device=device)
        self.v = torch.zeros(dim, device=device)
        self.t = 0

    def step(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the ascent step for gradient `grad`."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return self.lr * m_hat / (v_hat.sqrt() + self.eps)


class ClipUp:
    """Normalized-gradient momentum with a speed limit (Toklu et al. 2020)."""

    def __init__(self, dim, device, max_speed, step_size=None, momentum=0.9):
        self.max_speed = max_speed
        self.step_size = step_size if step_size is not None else max_speed / 2
        self.momentum = momentum
        self.velocity = torch.zeros(dim, device=device)

    def step(self, grad: torch.Tensor) -> torch.Tensor:
        g = grad / (grad.norm() + 1e-12)
        self.velocity = self.momentum * self.velocity + self.step_size * g
        speed = self.velocity.norm()
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        return self.velocity


class SGD:
    """Plain (optionally momentum) gradient ascent: step = lr * grad."""

    def __init__(self, dim, device, lr, momentum=0.0):
        self.lr, self.momentum = lr, momentum
        self.velocity = torch.zeros(dim, device=device)

    def step(self, grad: torch.Tensor) -> torch.Tensor:
        self.velocity = self.momentum * self.velocity + grad
        return self.lr * self.velocity


# Default step size per optimizer (Adam lr, SGD lr, ClipUp max speed).
DEFAULT_LR = {"adam": 0.01, "sgd": 0.01, "clipup": 0.015}


def make_optimizer(name: str, dim: int, device, lr: float | None):
    lr = lr if lr is not None else DEFAULT_LR[name]
    if name == "adam":
        return Adam(dim, device, lr)
    if name == "sgd":
        return SGD(dim, device, lr)
    if name == "clipup":
        return ClipUp(dim, device, max_speed=lr)
    raise ValueError(f"unknown optimizer {name}")


class Algorithm:
    name = "base"
    popsize: int
    mean: torch.Tensor

    def ask(self) -> torch.Tensor:
        raise NotImplementedError

    def tell(self, fitness: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError


class ARS(Algorithm):
    """ARS V2-t: antithetic directions, top-b selection, reward-std scaling."""

    name = "ars"

    def __init__(self, theta0, num_dirs=1024, top_dirs=256, step_size=0.02,
                 noise=0.02, seed=0):
        self.mean = theta0.clone()
        self.num_dirs, self.top_dirs = num_dirs, min(top_dirs, num_dirs)
        self.step_size, self.noise = step_size, noise
        self.popsize = 2 * num_dirs
        self.gen = torch.Generator(device=theta0.device).manual_seed(seed)

    def ask(self):
        self.delta = torch.randn(self.num_dirs, self.mean.numel(),
                                 device=self.mean.device, generator=self.gen)
        plus = self.mean + self.noise * self.delta
        minus = self.mean - self.noise * self.delta
        return torch.cat([plus, minus])

    def tell(self, fitness):
        r_plus, r_minus = fitness[: self.num_dirs], fitness[self.num_dirs :]
        best = torch.maximum(r_plus, r_minus).topk(self.top_dirs).indices
        sigma_r = torch.cat([r_plus[best], r_minus[best]]).std() + 1e-8
        diff = r_plus[best] - r_minus[best]
        step = self.step_size / (self.top_dirs * sigma_r) * (diff @ self.delta[best])
        self.mean += step
        return {"reward_std": sigma_r.item(), "step_norm": step.norm().item()}


class OpenAIES(Algorithm):
    """Antithetic Gaussian ES with centered ranks, Adam and weight decay."""

    name = "openai_es"

    def __init__(self, theta0, num_pairs=1024, noise=0.02, lr=None,
                 weight_decay=0.005, optimizer="adam", seed=0):
        self.mean = theta0.clone()
        self.num_pairs, self.noise, self.weight_decay = num_pairs, noise, weight_decay
        self.popsize = 2 * num_pairs
        self.optim = make_optimizer(optimizer, theta0.numel(), theta0.device, lr)
        self.gen = torch.Generator(device=theta0.device).manual_seed(seed)

    def ask(self):
        self.eps = torch.randn(self.num_pairs, self.mean.numel(),
                               device=self.mean.device, generator=self.gen)
        return torch.cat([self.mean + self.noise * self.eps,
                          self.mean - self.noise * self.eps])

    def tell(self, fitness):
        f = centered_ranks(fitness)
        f_plus, f_minus = f[: self.num_pairs], f[self.num_pairs :]
        grad = (f_plus - f_minus) @ self.eps / (2 * self.num_pairs * self.noise)
        grad = grad - self.weight_decay * self.mean
        step = self.optim.step(grad)
        self.mean += step
        return {"grad_norm": grad.norm().item(), "step_norm": step.norm().item()}


class PGPE(Algorithm):
    """PGPE with symmetric sampling, centered ranks and ClipUp.

    Search distribution: theta ~ N(mean, diag(sigma^2)). Both the mean and the
    per-parameter sigma are updated from the same rewards, so the method
    explores more along parameters where exploring pays off.
    """

    name = "pgpe"

    def __init__(self, theta0, num_pairs=1024, radius_init=None, sigma_init=None,
                 lr=None, sigma_lr=0.1, sigma_max_change=0.2, optimizer="clipup",
                 seed=0):
        d = theta0.numel()
        self.mean = theta0.clone()
        self.num_pairs = num_pairs
        self.popsize = 2 * num_pairs
        if sigma_init is None:
            # Toklu et al.: set the initial noise from a radius ||sigma||
            # about 15-18x ClipUp's max speed (0.27 for max speed 0.015).
            radius = radius_init if radius_init is not None else 0.27
            sigma_init = radius / math.sqrt(d)
        self.sigma = torch.full_like(theta0, sigma_init)
        self.optim = make_optimizer(optimizer, d, theta0.device, lr)
        self.sigma_lr, self.sigma_max_change = sigma_lr, sigma_max_change
        self.gen = torch.Generator(device=theta0.device).manual_seed(seed)

    def ask(self):
        self.eps = self.sigma * torch.randn(self.num_pairs, self.mean.numel(),
                                            device=self.mean.device,
                                            generator=self.gen)
        return torch.cat([self.mean + self.eps, self.mean - self.eps])

    def tell(self, fitness):
        f = centered_ranks(fitness)
        f_plus, f_minus = f[: self.num_pairs], f[self.num_pairs :]
        n = self.num_pairs
        grad_mean = (f_plus - f_minus) @ self.eps / (2 * n)
        baseline = f.mean()
        grad_sigma = (((f_plus + f_minus) / 2 - baseline)
                      @ ((self.eps**2 - self.sigma**2) / self.sigma)) / n
        step = self.optim.step(grad_mean)
        self.mean += step
        new_sigma = self.sigma + self.sigma_lr * grad_sigma
        lo = self.sigma * (1 - self.sigma_max_change)
        hi = self.sigma * (1 + self.sigma_max_change)
        self.sigma = torch.minimum(torch.maximum(new_sigma, lo), hi)
        return {"step_norm": step.norm().item(),
                "sigma_mean": self.sigma.mean().item(),
                "sigma_min": self.sigma.min().item(),
                "sigma_max": self.sigma.max().item()}


class CMAES(Algorithm):
    """(mu/mu_w, lambda)-CMA-ES, maximizing fitness.

    Full covariance (D x D) by default; `diagonal=True` gives sep-CMA-ES, which
    keeps only the diagonal and scales the learning rates up by (D+2)/3.
    """

    name = "cmaes"

    def __init__(self, theta0, popsize=None, sigma0=0.1, diagonal=False, seed=0):
        n = theta0.numel()
        dev = theta0.device
        self.n, self.diagonal = n, diagonal
        self.popsize = lam = popsize or 4 + int(3 * math.log(n))
        self.mu = mu = lam // 2
        w = math.log(mu + 0.5) - torch.log(torch.arange(1, mu + 1, dtype=torch.float64))
        self.weights = (w / w.sum()).to(dev)
        self.mueff = mueff = 1.0 / (self.weights**2).sum().item()

        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        if diagonal:
            self.c1 *= (n + 2) / 3
            self.cmu = min(1 - self.c1, self.cmu * (n + 2) / 3)
        self.damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        self.mean64 = theta0.double().clone()
        self.sigma = sigma0
        self.pc = torch.zeros(n, device=dev, dtype=torch.float64)
        self.ps = torch.zeros_like(self.pc)
        if diagonal:
            self.c = torch.ones_like(self.pc)
        else:
            self.C = torch.eye(n, device=dev, dtype=torch.float64)
            self.B = torch.eye(n, device=dev, dtype=torch.float64)
            self.D = torch.ones(n, device=dev, dtype=torch.float64)
            self.eigen_gen = 0
        self.generation = 0
        self.gen = torch.Generator(device=dev).manual_seed(seed)

    @property
    def mean(self):
        return self.mean64.float()

    def ask(self):
        z = torch.randn(self.popsize, self.n, device=self.mean64.device,
                        dtype=torch.float64, generator=self.gen)
        if self.diagonal:
            self.y = z * self.c.sqrt()
        else:
            self.y = (z * self.D) @ self.B.T  # rows: B @ (D * z)
        return (self.mean64 + self.sigma * self.y).float()

    def tell(self, fitness):
        n, cs, cc, c1, cmu = self.n, self.cs, self.cc, self.c1, self.cmu
        order = fitness.argsort(descending=True)[: self.mu]
        y_sel = self.y[order]
        y_w = self.weights @ y_sel
        self.mean64 += self.sigma * y_w

        # Step-size path, in coordinates where the distribution is isotropic.
        if self.diagonal:
            inv_sqrt_c_yw = y_w / self.c.sqrt()
        else:
            inv_sqrt_c_yw = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = (1 - cs) * self.ps + math.sqrt(cs * (2 - cs) * self.mueff) * inv_sqrt_c_yw
        ps_norm = self.ps.norm().item()
        self.generation += 1
        hsig = (ps_norm / math.sqrt(1 - (1 - cs) ** (2 * self.generation)) / self.chi_n
                < 1.4 + 2 / (n + 1))
        self.pc = (1 - cc) * self.pc + hsig * math.sqrt(cc * (2 - cc) * self.mueff) * y_w

        # Covariance: rank-one (evolution path) + rank-mu (selected steps).
        old_scale = 1 - c1 - cmu + (0 if hsig else c1 * cc * (2 - cc))
        if self.diagonal:
            self.c = (old_scale * self.c + c1 * self.pc**2
                      + cmu * (self.weights @ y_sel**2))
        else:
            self.C = (old_scale * self.C + c1 * torch.outer(self.pc, self.pc)
                      + cmu * (y_sel.T * self.weights) @ y_sel)
            # Eigendecomposition is O(n^3): only redo it when C has changed
            # appreciably (Hansen's lazy-update schedule), but at least every
            # 25 generations: for n in the thousands the lazy rule would wait
            # ~1000 generations, so a short run would never sample from the
            # covariance it has learned.
            lazy = self.popsize / (c1 + cmu) / n / 10
            if self.generation - self.eigen_gen > min(lazy, 25):
                self.eigen_gen = self.generation
                self.C = torch.triu(self.C) + torch.triu(self.C, 1).T
                evals, self.B = torch.linalg.eigh(self.C)
                self.D = evals.clamp_min(1e-20).sqrt()

        self.sigma *= math.exp((cs / self.damps) * (ps_norm / self.chi_n - 1))
        scale = self.c.sqrt() if self.diagonal else self.D
        return {"sigma": self.sigma,
                "axis_ratio": (scale.max() / scale.min()).item(),
                "ps_norm_over_chi": ps_norm / self.chi_n}


def make_algorithm(name: str, theta0: torch.Tensor, args) -> Algorithm:
    """Build an algorithm from train_es.py's command-line arguments."""
    pairs = args.popsize // 2
    if name == "ars":
        return ARS(theta0, num_dirs=pairs, top_dirs=args.top_dirs or pairs // 4,
                   step_size=args.lr or 0.02, noise=args.sigma or 0.02,
                   seed=args.seed)
    if name == "openai_es":
        return OpenAIES(theta0, num_pairs=pairs, noise=args.sigma or 0.02,
                        lr=args.lr, weight_decay=args.weight_decay,
                        optimizer=args.optimizer or "adam", seed=args.seed)
    if name == "pgpe":
        return PGPE(theta0, num_pairs=pairs, sigma_init=args.sigma, lr=args.lr,
                    optimizer=args.optimizer or "clipup", seed=args.seed)
    if name in ("cmaes", "sep_cmaes"):
        return CMAES(theta0, popsize=args.popsize, sigma0=args.sigma or 0.1,
                     diagonal=name == "sep_cmaes", seed=args.seed)
    raise ValueError(f"unknown algorithm {name}")

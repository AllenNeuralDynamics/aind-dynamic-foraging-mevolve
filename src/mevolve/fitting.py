"""Fit a program's parameters to one subject: per subject (MLE), or as a 2-level model
(session -> subject) fitted by Laplace-EM.

Every bounded parameter is fitted on an unconstrained scale, ``low + (high - low) * Phi(x)``, so a
standard normal on ``x`` is uniform on the parameter's bounds, as in study 13's hierarchical models.
Sessions are padded into one array per subject and masked, so a fit is one jitted L-BFGS run.

- **Per subject**: one parameter set for all of a subject's fit sessions, by maximum likelihood.
- **2-level** (session -> subject): each fit session's parameters ``theta_s ~ N(mu, sigma^2)`` vary around the
  subject's typical values ``mu``, with study 13's fixed hyperprior ``mu ~ N(0, 1)`` and
  ``log sigma ~ N(-1, 1)``. ``mu`` and ``sigma`` are fitted by Laplace-EM (the hierarchical EM of
  Huys et al. 2011), which maximizes the likelihood with each session's parameters integrated out.
  A joint MAP of (mu, sigma, theta_s) does not work: in non-centred form nothing resists sigma
  growing (on study 13's test subjects it gave spreads about 15 times study 13's posterior and a
  session-marginal weight ESS near 1), and in centred form sigma collapses to 0 (MEvolve ADR 0004).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.stats import norm

from .program import Program, SESSION_FACTS

LOG_SIGMA_PRIOR = (-1.0, 1.0)   # mean and sd of log sigma (study 13's fixed hyperprior)
ABANDON_NATS = 100.0            # multi-start: drop a start this far behind the best so far
FACE_POINTS = 64               # Sobol points screened on each bound face


@dataclass(frozen=True)
class Batch:
    """One subject's sessions, padded to (n_sessions, n_trials) and masked."""
    choices: jnp.ndarray
    rewards: jnp.ndarray
    mask: jnp.ndarray
    facts: dict

    @property
    def n_trials(self) -> int:
        return int(self.mask.sum())


def _bucket(n: int, step: int) -> int:
    return max(step, -(-n // step) * step)


def pad(sessions, trial_step: int = 128, session_step: int = 4) -> Batch:
    """Pad sessions (objects with ``choices``, ``rewards`` and session facts) into a ``Batch``.

    Sizes are rounded up to buckets so that subjects of similar size share one compilation.
    Padded trials come after each session's real ones, so they never affect a prediction.
    """
    n_s = _bucket(len(sessions), session_step)
    n_t = _bucket(max(len(s.choices) for s in sessions), trial_step)
    choices = np.zeros((n_s, n_t), np.int32)
    rewards = np.zeros((n_s, n_t), np.float64)
    mask = np.zeros((n_s, n_t), bool)
    facts = {k: np.zeros(n_s) for k in SESSION_FACTS}
    for i, s in enumerate(sessions):
        n = len(s.choices)
        choices[i, :n], rewards[i, :n], mask[i, :n] = s.choices, s.rewards, True
        for k, v in session_facts(s).items():
            facts[k][i] = v
    return Batch(jnp.asarray(choices), jnp.asarray(rewards), jnp.asarray(mask),
                 {k: jnp.asarray(v) for k, v in facts.items()})


def session_facts(session) -> dict[str, float]:
    """The facts a program receives at a session's start: stage group (-1 unknown, 0 earlier,
    1 final), coupled and baited task (1 or 0) and the session's index in the subject's history."""
    task = getattr(session, "task", None) or ""
    stage = {"unknown": -1.0, "earlier": 0.0, "final": 1.0}[getattr(session, "stage_group", "unknown")]
    return {"stage": stage, "coupled": float(task.startswith("Coupled")),
            "baited": float("Baiting" in task and "Without" not in task),
            "index": float(getattr(session, "index", 0))}


def bounds_arrays(program: Program) -> tuple[jnp.ndarray, jnp.ndarray]:
    lo, hi = zip(*program.params.values())
    return jnp.asarray(lo), jnp.asarray(hi)


def to_params(program: Program, x: jnp.ndarray) -> dict:
    """Unconstrained vector(s) ``x[..., n_params]`` -> bounded parameters by name."""
    lo, hi = bounds_arrays(program)
    v = lo + (hi - lo) * norm.cdf(x)
    return {k: v[..., i] for i, k in enumerate(program.params)}


def from_params(program: Program, p: dict) -> jnp.ndarray:
    lo, hi = bounds_arrays(program)
    u = (jnp.asarray([p[k] for k in program.params]) - lo) / (hi - lo)
    return norm.ppf(jnp.clip(u, 1e-12, 1 - 1e-12))


def trial_logp(program: Program, p: dict, choices, rewards, facts) -> jnp.ndarray:
    """log p(observed choice) on each trial of one session, unclipped (log-softmax of the logits)."""
    def step(latents, trial):
        choice, reward = trial
        logp = jax.nn.log_softmax(jnp.asarray(program.predict(latents, p), dtype=jnp.float64))
        return program.update(latents, choice, reward, p), logp[choice]

    latents0 = jnp.asarray(program.init_latents(p, facts), dtype=jnp.float64)
    return jax.lax.scan(step, latents0, (choices, rewards))[1]


def batch_logp(program: Program, x: jnp.ndarray, batch: Batch) -> jnp.ndarray:
    """Per-trial log p for every session, shape (n_sessions, n_trials); ``x`` is one unconstrained
    vector for all sessions (n_params,) or one per session (n_sessions, n_params)."""
    xs = jnp.broadcast_to(x, (batch.choices.shape[0], x.shape[-1]))
    facts = {k: batch.facts[k] for k in SESSION_FACTS}

    def one(xi, c, r, f):
        return trial_logp(program, to_params(program, xi), c, r, f)

    return jax.vmap(one)(xs, batch.choices, batch.rewards, facts)


def lbfgs(fun, x0, max_iter: int = 300, tol: float = 1e-7, abandon: float = jnp.inf, patience: int = 30):
    """Minimize ``fun`` from ``x0`` with optax's L-BFGS and zoom line search; returns (x, value).

    A start still above ``abandon`` after ``patience`` iterations stops early: in a multi-start fit
    it cannot win, and one such start would otherwise run all ``max_iter`` iterations.
    """
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(fun)

    def step(carry):
        x, state, _, _, i = carry
        value, grad = value_and_grad(x, state=state)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=fun)
        return optax.apply_updates(x, updates), state, jnp.linalg.norm(grad), value, i + 1

    def keep_going(carry):
        _, _, gnorm, value, i = carry
        return (i == 0) | ((i < max_iter) & (gnorm > tol) & ((i < patience) | (value <= abandon)))

    x, _, _, _, _ = jax.lax.while_loop(keep_going, step, (x0, opt.init(x0), jnp.inf, jnp.inf, 0))
    return x, fun(x)


ZERO_LOGP = float(np.log(1e-300))   # programs write a zero probability as 1e-300 or below
REF_ZERO_LOGP = float(np.log(1e-16))  # the reference fit's negLL charges a zero as 1e-16


def _nll(program: Program, x, choices, rewards, mask, facts):
    """The fit objective, as the reference fit's ``negLL``: an exact zero probability costs
    -log(1e-16); small positive probabilities are kept as they are."""
    batch = Batch(choices, rewards, mask, facts)
    logp = batch_logp(program, x, batch)
    logp = jnp.where(logp <= ZERO_LOGP, REF_ZERO_LOGP, logp)
    return -jnp.sum(jnp.where(mask, logp, 0.0))


def per_subject_nll(program: Program, batch: Batch):
    """The per-subject objective: negative log likelihood of all fit sessions at one parameter set."""
    return lambda x: _nll(program, x, batch.choices, batch.rewards, batch.mask, batch.facts)


_SOLVERS: dict = {}


def _solver(program: Program, kind: str, max_iter: int):
    """A jitted L-BFGS solve for one program, compiled once per array shape; the data are
    arguments, not closed-over constants. One start or one session per call, in a Python loop:
    under vmap or lax.map the nested line-search loops ran 20 times slower."""
    key = (id(program), kind, max_iter)
    if key not in _SOLVERS:
        if kind == "subject":
            def solve(x0, c, r, m, f, abandon):
                return lbfgs(lambda x: _nll(program, x, c, r, m, f), x0, max_iter, abandon=abandon)
        elif kind == "screen":
            def solve(xs, c, r, m, f):
                return jax.vmap(lambda x: _nll(program, x, c, r, m, f))(xs)
        elif kind == "value":
            def solve(x, c, r, m, f):
                return _nll(program, x, c, r, m, f)
        else:
            def solve(t0, c, r, m, f, mu, sigma):
                def fun(t):
                    logp = trial_logp(program, to_params(program, t), c, r, f)
                    return -jnp.sum(jnp.where(m, logp, 0.0)) + 0.5 * jnp.sum(((t - mu) / sigma) ** 2)
                t, _ = lbfgs(fun, t0, max_iter)
                h = jax.hessian(fun)(t)
                w, v = jnp.linalg.eigh(0.5 * (h + h.T))
                cov = (v / jnp.maximum(w, 1e-8)) @ v.T      # a guard where the Hessian is not positive
                return t, jnp.diag(cov)
        _SOLVERS[key] = jax.jit(solve)
    return _SOLVERS[key]


def fit_per_subject(program: Program, batch: Batch, n_starts: int = 8, seed: int = 0, max_iter: int = 300,
                    de_restarts: int = 4, de_generations: int = 40, polish: bool = True):
    """MLE of one parameter set, in three steps.

    1. Differential evolution on the bounded scale, as the reference fits (study 12):
       ``de_restarts`` independent runs of a fixed ``de_generations`` each, with no tolerance
       stop; each generation's population (15 per parameter) is evaluated in one batched pass.
    2. L-BFGS from the best ``n_starts / de_restarts`` (at least one) members of each run's final population,
       and from the best point on each bound face (one parameter pinned at a bound; see below).
    3. Nelder-Mead from the best result.

    Gradients alone failed gate 1b where a likelihood is flat in some parameters almost
    everywhere (Rescorla-Wagner's epsilon-greedy choice) or has several optima
    (FeedbackDependent-RL without a bias); scipy's default DE tolerance stopped after about 20
    generations, still short of the reference on Rescorla-Wagner. One DE run settles in one basin:
    on Miller-RHG's 16 failing subjects (issue #17), 400 generations instead of 150 fixed 1 and 16
    L-BFGS starts fixed none, while 4 independent runs of 40 generations fixed 15.

    The bound faces catch an optimum on a bound inside a narrow basin, which DE's interior draws
    rarely sample and L-BFGS cannot reach, since a bound is at infinity on its probit scale.
    ActorCritic's critic learning rate has one at 0, about 0.003 wide, 1,000 nats better than
    the interior optimum on some subjects (issue #15). Each face is screened over the DE starts
    and ``FACE_POINTS`` Sobol points with the parameter pinned, in one batched pass.

    Returns the best unconstrained vector, its parameters and its negative log likelihood.
    """
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import qmc

    n = len(program.params)
    data = (batch.choices, batch.rewards, batch.mask, batch.facts)
    screen = _solver(program, "screen", 0)
    lo, hi = (np.asarray(a) for a in bounds_arrays(program))

    def to_x(v):        # bounded (n, ...) -> unconstrained (..., n)
        return norm.ppf(np.clip((np.moveaxis(v, 0, -1) - lo) / (hi - lo), 1e-12, 1 - 1e-12))

    def population_nll(v):
        out = np.asarray(screen(jnp.asarray(to_x(v)), *data))
        return np.where(np.isfinite(out), out, 1e12)

    top = []
    for restart in range(de_restarts):
        de = differential_evolution(population_nll, list(zip(lo, hi)), popsize=15, maxiter=de_generations, tol=0,
                                    atol=0, seed=seed + restart, polish=False, vectorized=True, updating="deferred",
                                    init="sobol")
        order = np.argsort(de.population_energies, kind="stable")[:max(1, n_starts // de_restarts)]
        top.append(de.population[order])
    top = np.vstack(top)
    faces = []
    for j in range(n):
        for bound in (lo[j], hi[j]):
            v = np.vstack([top, lo + (hi - lo) * qmc.Sobol(n, seed=seed + j).random(FACE_POINTS)])
            v[:, j] = bound
            faces.append(v[np.argmin(population_nll(v.T))])
    starts = jnp.asarray(to_x(np.vstack([top] + faces).T))
    solve = _solver(program, "subject", max_iter)
    out, best = [], np.inf
    for x0 in starts:      # a start 100 nats behind the best so far after 30 iterations is dropped
        x, value = solve(x0, batch.choices, batch.rewards, batch.mask, batch.facts, best + ABANDON_NATS)
        out.append((x, value))
        if np.isfinite(float(value)):
            best = min(best, float(value))
    xs = jnp.stack([o[0] for o in out])
    values = jnp.stack([o[1] for o in out])
    values = jnp.where(jnp.isfinite(values), values, jnp.inf)
    n_top = len(top)
    best = int(jnp.argmin(values))
    if polish:
        value_fn = _solver(program, "value", 0)
        f = lambda x: float(value_fn(jnp.asarray(x), *data))
        res = minimize(f, np.asarray(xs[best]), method="Nelder-Mead",
                       options={"maxfev": 200 * n, "xatol": 1e-6, "fatol": 1e-7})
        if np.isfinite(res.fun) and res.fun < float(values[best]):
            xs = xs.at[best].set(jnp.asarray(res.x))
            values = values.at[best].set(res.fun)
    return {"x": xs[best], "params": {k: float(v) for k, v in to_params(program, xs[best]).items()},
            "nll": float(values[best]), "nll_starts": np.asarray(values[:n_top]),
            "nll_faces": np.asarray(values[n_top:])}


def _session_map(program: Program, batch: Batch, mu, sigma, theta0, max_iter: int):
    """E-step: each live session's MAP under N(mu, sigma^2), and the diagonal of its Laplace
    covariance. Padded sessions keep their start and a zero variance."""
    solve = _solver(program, "session", max_iter)
    live = np.asarray(batch.mask.any(axis=1))
    theta, var = np.array(theta0), np.zeros(theta0.shape)
    for i in np.flatnonzero(live):
        f = {k: batch.facts[k][i] for k in SESSION_FACTS}
        t, v = solve(theta0[i], batch.choices[i], batch.rewards[i], batch.mask[i], f, mu, sigma)
        theta[i], var[i] = np.asarray(t), np.asarray(v)
    return jnp.asarray(theta), jnp.asarray(var)


def _m_step(theta, var, live, log_sigma):
    """M-step: mu (Gaussian prior, closed form) and log sigma (log-normal prior, 1-D Newton)."""
    n = live.sum()
    sigma2 = jnp.exp(2 * log_sigma)
    mu = jnp.sum(jnp.where(live[:, None], theta, 0.0), axis=0) / (n + sigma2)     # mu ~ N(0, 1)
    a = jnp.sum(jnp.where(live[:, None], (theta - mu) ** 2 + var, 0.0), axis=0)
    m, s = LOG_SIGMA_PRIOR
    # maximize -n u - a exp(-2u) / 2 - (u - m)^2 / (2 s^2), concave in u: its derivative falls
    # strictly from +inf to -inf, so bisection finds the one root (Newton steps overshoot when a
    # is small)
    lo, hi = jnp.full_like(log_sigma, -30.0), jnp.full_like(log_sigma, 10.0)
    for _ in range(80):
        u = 0.5 * (lo + hi)
        rising = -n + a * jnp.exp(-2 * u) - (u - m) / s ** 2 > 0
        lo, hi = jnp.where(rising, u, lo), jnp.where(rising, hi, u)
    return mu, 0.5 * (lo + hi)


def fit_two_level(program: Program, batch: Batch, x_start: jnp.ndarray, max_em: int = 20,
                  tol: float = 1e-3, inner_iter: int = 50):
    """Typical values ``mu`` and log spreads by Laplace-EM, started from a per-subject fit.

    EM converges linearly, but the held-out score settles long before the spreads of poorly
    identified parameters do: on two of the slowest test subjects, 10 iterations scored within
    2e-5 of 100. Hence at most 20.
    """
    n_s, n = batch.choices.shape[0], len(program.params)
    live = batch.mask.any(axis=1)

    m_step = jax.jit(_m_step)

    def em_step(mu, log_sigma, theta):
        theta, var = _session_map(program, batch, mu, jnp.exp(log_sigma), theta, inner_iter)
        mu_new, log_sigma_new = m_step(theta, var, live, log_sigma)
        return mu_new, log_sigma_new, theta

    mu, log_sigma = jnp.asarray(x_start), jnp.full(n, LOG_SIGMA_PRIOR[0])
    theta = jnp.broadcast_to(mu, (n_s, n))
    for it in range(max_em):
        mu_new, log_sigma_new, theta = em_step(mu, log_sigma, theta)
        change = float(jnp.maximum(jnp.max(jnp.abs(mu_new - mu)), jnp.max(jnp.abs(log_sigma_new - log_sigma))))
        mu, log_sigma = mu_new, log_sigma_new
        if change < tol:
            break
    return {"mu": mu, "log_sigma": log_sigma, "theta": theta, "em_iterations": it + 1,
            "converged": change < tol,
            "typical_params": {k: float(v) for k, v in to_params(program, mu).items()}}

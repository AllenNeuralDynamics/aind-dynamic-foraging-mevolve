"""Fit a program's parameters to one subject: per subject (MLE) or at the session level (MAP).

Every bounded parameter is fitted on an unconstrained scale, ``low + (high - low) * Phi(x)``, so a
standard normal on ``x`` is uniform on the parameter's bounds, as in study 13's hierarchical models.
Sessions are padded into one array per subject and masked, so a fit is one jitted L-BFGS run.

- **Per subject**: one parameter set for all of a subject's fit sessions, by maximum likelihood.
- **Session level**: each fit session's parameters ``theta_s = mu + sigma * z_s`` vary around the
  subject's typical values ``mu`` (non-centred form), with ``mu ~ N(0, 1)``, ``log sigma ~ N(-1, 1)``
  and ``z_s ~ N(0, 1)``, study 13's fixed hyperprior; fitted jointly by MAP.
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


def lbfgs(fun, x0, max_iter: int = 300, tol: float = 1e-7):
    """Minimize ``fun`` from ``x0`` with optax's L-BFGS and zoom line search; returns (x, value)."""
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(fun)

    def step(carry):
        x, state, _, i = carry
        value, grad = value_and_grad(x, state=state)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=fun)
        return optax.apply_updates(x, updates), state, jnp.linalg.norm(grad), i + 1

    def keep_going(carry):
        _, _, gnorm, i = carry
        return (i == 0) | ((i < max_iter) & (gnorm > tol))

    x, _, _, _ = jax.lax.while_loop(keep_going, step, (x0, opt.init(x0), jnp.inf, 0))
    return x, fun(x)


def per_subject_nll(program: Program, batch: Batch):
    """The per-subject objective: negative log likelihood of all fit sessions at one parameter set."""
    def fun(x):
        return -jnp.sum(jnp.where(batch.mask, batch_logp(program, x, batch), 0.0))
    return fun


def fit_per_subject(program: Program, batch: Batch, n_starts: int = 8, seed: int = 0, max_iter: int = 300):
    """MLE of one parameter set; starts at the centre of the box plus ``n_starts - 1`` random ones.

    Returns the best unconstrained vector, its parameters and its negative log likelihood.
    """
    fun = per_subject_nll(program, batch)
    n = len(program.params)
    starts = jnp.concatenate([jnp.zeros((1, n)), jax.random.normal(jax.random.PRNGKey(seed), (n_starts - 1, n))])
    xs, values = jax.jit(jax.vmap(lambda x0: lbfgs(fun, x0, max_iter)))(starts)
    values = jnp.where(jnp.isfinite(values), values, jnp.inf)
    best = int(jnp.argmin(values))
    return {"x": xs[best], "params": {k: float(v) for k, v in to_params(program, xs[best]).items()},
            "nll": float(values[best]), "nll_starts": np.asarray(values)}


def session_level_objective(program: Program, batch: Batch):
    """Negative log posterior of (mu, log sigma, z) for one subject, in one flat vector."""
    n = len(program.params)
    n_s = batch.choices.shape[0]
    live = batch.mask.any(axis=1)          # padded sessions carry no data and no prior term
    m, s = LOG_SIGMA_PRIOR

    def unpack(v):
        return v[:n], v[n:2 * n], v[2 * n:].reshape(n_s, n)

    def fun(v):
        mu, log_sigma, z = unpack(v)
        theta = mu + jnp.exp(log_sigma) * z
        ll = jnp.sum(jnp.where(batch.mask, batch_logp(program, theta, batch), 0.0))
        prior = (jnp.sum(norm.logpdf(mu)) + jnp.sum(norm.logpdf(log_sigma, m, s))
                 + jnp.sum(jnp.where(live[:, None], norm.logpdf(z), 0.0)))
        return -(ll + prior)

    return fun, unpack


def fit_session_level(program: Program, batch: Batch, x_start: jnp.ndarray, max_iter: int = 500):
    """MAP of typical values, spreads and session offsets, started from a per-subject fit."""
    fun, unpack = session_level_objective(program, batch)
    n = len(program.params)
    v0 = jnp.concatenate([x_start, jnp.full(n, LOG_SIGMA_PRIOR[0]), jnp.zeros(batch.choices.shape[0] * n)])
    v, value = jax.jit(lambda v0: lbfgs(fun, v0, max_iter))(v0)
    mu, log_sigma, z = unpack(v)
    return {"mu": mu, "log_sigma": log_sigma, "z": z, "neg_log_posterior": float(value),
            "typical_params": {k: float(val) for k, val in to_params(program, mu).items()}}

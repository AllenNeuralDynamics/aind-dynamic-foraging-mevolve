"""Score a fitted program on a subject's held-out (score) sessions.

Per-trial probabilities are clipped to [1e-10, 1 - 1e-10] before the log, as the wrapper scores
every comparator. A study score pools sessions: exp(sum of log likelihoods / number of trials).

- **Point score**: one parameter set (a per-subject fit).
- **Session marginal**: a held-out session's own parameters were never observed, so they are
  averaged out: K draws ``theta_k = mu + sigma * eps_k`` from the subject's distribution, and the
  session scores ``log mean_k prod_t p_k(c_t)``. Averaging whole-session likelihoods before the
  log is the same as reweighting the draws, trial by trial, by how well they explained the
  session's earlier trials. ``eps`` is fixed per subject (common random draws), so two programs
  are compared on the same draws.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .fitting import Batch, batch_logp
from .program import Program

CLIP = 1e-10


def clipped(logp: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.clip(jnp.exp(logp), CLIP, 1.0 - CLIP))


def point_scores(program: Program, x: jnp.ndarray, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    """Per-session summed clipped log likelihood and trial count at one parameter set."""
    logp = clipped(batch_logp(program, x, batch))
    return (np.asarray(jnp.sum(jnp.where(batch.mask, logp, 0.0), axis=1)),
            np.asarray(batch.mask.sum(axis=1)))


def marginal_scores(program: Program, mu: jnp.ndarray, log_sigma: jnp.ndarray, batch: Batch,
                    key: jax.Array, n_draws: int = 256) -> dict[str, np.ndarray]:
    """Per-session session-marginal log likelihood, trial count and weight ESS.

    The ESS of the final weights (proportional to each draw's whole-session likelihood) says how
    many draws effectively carry the session; a value near 1 means the marginal is noisy.
    """
    eps = jax.random.normal(key, (n_draws, mu.shape[0]))
    theta = mu + jnp.exp(log_sigma) * eps

    def one_draw(t):
        logp = clipped(batch_logp(program, t, batch))
        return jnp.sum(jnp.where(batch.mask, logp, 0.0), axis=1)      # (n_sessions,)

    session_ll = jax.lax.map(one_draw, theta)                            # (n_draws, n_sessions)
    log_marginal = logsumexp(session_ll, axis=0) - jnp.log(n_draws)
    w = jnp.exp(session_ll - session_ll.max(axis=0))
    ess = w.sum(axis=0) ** 2 / (w ** 2).sum(axis=0)
    return {"log_lik": np.asarray(log_marginal), "n_trials": np.asarray(batch.mask.sum(axis=1)),
            "ess": np.asarray(ess)}


def pooled(log_lik, n_trials) -> float:
    """exp(sum log likelihood / sum trials) over any set of sessions or subjects."""
    return float(np.exp(np.sum(log_lik) / np.sum(n_trials)))

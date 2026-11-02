"""The fitter: fit a program to one subject's sessions under one fitting regime.

    regime "subject": one parameter set for all sessions, by maximum likelihood
    regime "2level":  the 2-level model (session -> subject) fitted by Laplace-EM, started from the
                      per-subject fit (ADR 0004): typical values, spreads, each session's parameters

``fit`` returns the parameters; ``Fit.latents`` replays the fitted sessions and returns the latent
variables and choice probabilities on every trial. The fitter knows nothing of scoring held-out
sessions or of the search: that is the evaluator's use of it (``mevolve.evaluate``). With the
program interface (``mevolve.program``) and the numerical core (``mevolve.fitting``) it is the part
that moves to ``aind-dynamic-foraging-models`` (ADR 0005).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import fitting
from .program import Program, replay

REGIMES = ("subject", "2level")


@dataclass
class Fit:
    """One subject's fit. ``x`` is the per-subject fit on the unconstrained scale; for 2-level,
    ``mu`` and ``log_sigma`` are the typical values and log spreads on that scale and
    ``session_x`` each session's parameters (the E-step's MAP)."""
    program: Program
    regime: str
    sessions: list
    x: jnp.ndarray
    params: dict
    nll: float
    mu: jnp.ndarray | None = None
    log_sigma: jnp.ndarray | None = None
    session_x: jnp.ndarray | None = None
    typical_params: dict | None = None
    em_iterations: int | None = None
    converged: bool | None = None
    batch: fitting.Batch | None = field(default=None, repr=False)

    def session_params(self) -> list[dict]:
        """Each fitted session's parameters: the per-subject set, or the session's own (2-level)."""
        xs = self.session_x if self.regime == "2level" else jnp.broadcast_to(self.x, (len(self.sessions), len(self.x)))
        return [{k: float(v) for k, v in fitting.to_params(self.program, xi).items()} for xi in xs[: len(self.sessions)]]

    def latents(self) -> list[dict]:
        """Latent variables before each trial and the choice probabilities, for every fitted
        session at that session's parameters: ``{"latents": (n_trials, N_LATENT), "probs":
        (n_trials, 2)}`` per session, in input order."""
        out = []
        for s, p in zip(self.sessions, self.session_params()):
            probs, lat = _replay(self.program)(p, jnp.asarray(s.choices, jnp.int32),
                                               jnp.asarray(s.rewards, jnp.float64), fitting.session_facts(s))
            out.append({"latents": np.asarray(lat), "probs": np.asarray(probs)})
        return out


_REPLAYS: dict = {}


def _replay(program: Program):
    if id(program) not in _REPLAYS:
        _REPLAYS[id(program)] = jax.jit(lambda p, c, r, f: replay(program, p, c, r, f))
    return _REPLAYS[id(program)]


def fit(program: Program, sessions: list, regime: str = "subject", *, seed: int = 0, n_starts: int = 8,
        max_em: int = 20) -> Fit:
    """Fit ``program`` to ``sessions`` (objects with ``choices``, ``rewards`` and session facts).

    ``max_em`` caps Laplace-EM: 20 for the search, 100 to run it to convergence for reported
    scores (ADR 0004). The per-subject fit uses the fitter's default budget either way.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, not {regime!r}")
    batch = fitting.pad(sessions)
    per = fitting.fit_per_subject(program, batch, n_starts=n_starts, seed=seed)
    result = Fit(program, regime, list(sessions), per["x"], per["params"], per["nll"], batch=batch)
    if regime == "2level":
        sl = fitting.fit_two_level(program, batch, per["x"], max_em=max_em)
        result.mu, result.log_sigma, result.session_x = sl["mu"], sl["log_sigma"], sl["theta"]
        result.typical_params = sl["typical_params"]
        result.em_iterations, result.converged = sl["em_iterations"], bool(sl["converged"])
    return result

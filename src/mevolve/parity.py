"""Gate 1a: a literature program reproduces its reference forager, trial by trial.

The reference is the numpy forager of ``aind-dynamic-foraging-models`` (``literature.oracle``),
replayed with ``perform_closed_loop_multi_session`` on the same sessions (ignored trials removed).
Pass: every per-trial choice probability within 1e-6, and the pooled likelihood within 1e-5.
The pooled likelihood clips p to [1e-10, 1 - 1e-10], as the wrapper scores every comparator.

One reference behaviour cannot be matched by any deterministic program: when a logit exceeds 700,
``act_functions.softmax`` returns a one-hot choice at a random side (it takes the argmax of a
shuffled copy of the logits and uses that index in the unshuffled array). Hattori2019, Bari2019
and ActorCritic choose through it; only ActorCritic, whose preferences are unbounded, gets there.
From the first such trial to the end of its session, trials are left out of both comparisons and
counted (``n_excluded``): the model learns from the choice probability it used, so one random trial
changes every later one.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import numpy as np

from . import literature, program

PER_TRIAL_TOL = 1e-6
POOLED_TOL = 1e-5
SOFTMAX_LIMIT = 700.0
REFERENCE_SOFTMAX = {"Hattori2019", "Bari2019", "ActorCritic"}   # choose via act_functions.softmax


@lru_cache(maxsize=None)
def _program(name: str):
    """The program and a jitted replay that also returns the largest logit on each trial."""
    prog = program.literature(literature.LITERATURE[name][0])

    @jax.jit
    def run(p, choices, rewards):
        probs, latents = program.replay(prog, p, choices, rewards)
        logits = jax.vmap(lambda lat: prog.predict(lat, p))(latents)
        return probs, logits.max(axis=1)

    return prog, run


def pooled_likelihood(probs: list[np.ndarray], choices: list[np.ndarray], keep: list[np.ndarray] | None = None) -> float:
    """exp(mean log p(choice)) over all (kept) trials, p clipped as the wrapper clips it."""
    keep = keep or [np.ones(len(c), bool) for c in choices]
    logp = [np.log(np.clip(p[np.arange(len(c)), c], 1e-10, 1 - 1e-10))[k] for p, c, k in zip(probs, choices, keep)]
    return float(np.exp(np.concatenate(logp).mean()))


def draw_params(name: str, rng: np.random.Generator) -> dict[str, float]:
    """Free parameters drawn uniformly within the reference fit's bounds."""
    free, _ = literature.fit_spec(name)
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in free.items()}


def compare(name: str, params: dict[str, float], choices: list[np.ndarray], rewards: list[np.ndarray]) -> dict:
    """Replay one literature model both ways on the same sessions, at the same parameters."""
    _, clamped = literature.fit_spec(name)
    ref = literature.oracle(name, params={**params, **clamped})
    ref_probs = [np.asarray(p).T for p in ref.perform_closed_loop_multi_session(choices, rewards)]
    _, run = _program(name)
    p = {k: np.float64(v) for k, v in params.items()}
    out = [run(p, c.astype(np.int32), r.astype(np.float64)) for c, r in zip(choices, rewards)]
    prog_probs = [np.asarray(o[0]) for o in out]
    # From the first trial past the limit to the end of the session: ActorCritic learns from the
    # choice probability it used, so one random trial changes every later one.
    keep = [np.ones(len(c), bool) if name not in REFERENCE_SOFTMAX
            else np.cumsum(np.asarray(o[1]) > SOFTMAX_LIMIT) == 0 for c, o in zip(choices, out)]
    diff = max((float(np.max(np.abs(a - b)[k])) if k.any() else 0.0) for a, b, k in zip(ref_probs, prog_probs, keep))
    pooled_ref = pooled_likelihood(ref_probs, choices, keep)
    pooled_prog = pooled_likelihood(prog_probs, choices, keep)
    return {"max_abs_diff": diff, "pooled_reference": pooled_ref, "pooled_program": pooled_prog,
            "pooled_diff": abs(pooled_ref - pooled_prog), "n_trials": int(sum(len(c) for c in choices)),
            "n_excluded": int(sum((~k).sum() for k in keep)),
            "ok": diff <= PER_TRIAL_TOL and abs(pooled_ref - pooled_prog) <= POOLED_TOL}


def synthetic_sessions(rng: np.random.Generator, n_sessions: int = 3, n_trials: int = 150):
    """Choices with streaks and switches, rewards at about 40%: enough to exercise every branch."""
    choices, rewards = [], []
    for _ in range(n_sessions):
        c = np.zeros(n_trials, dtype=np.int64)
        for t in range(1, n_trials):
            c[t] = c[t - 1] if rng.random() < 0.8 else 1 - c[t - 1]
        choices.append(c)
        rewards.append((rng.random(n_trials) < 0.4).astype(np.float64))
    return choices, rewards

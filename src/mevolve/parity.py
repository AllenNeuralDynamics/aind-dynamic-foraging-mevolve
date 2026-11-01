"""Gate 1a: a literature program reproduces its reference forager, trial by trial.

The reference is the numpy forager of ``aind-dynamic-foraging-models`` (``literature.oracle``),
replayed with ``perform_closed_loop_multi_session`` on the same sessions (ignored trials removed).
Pass: every per-trial choice probability within 1e-6, and the pooled likelihood within 1e-5.
The pooled likelihood clips p to [1e-10, 1 - 1e-10], as the wrapper scores every comparator.
"""
from __future__ import annotations

import numpy as np

from . import literature, program

PER_TRIAL_TOL = 1e-6
POOLED_TOL = 1e-5


def pooled_likelihood(probs: list[np.ndarray], choices: list[np.ndarray]) -> float:
    """exp(mean log p(choice)) over all trials, p clipped as the wrapper clips it."""
    logp = [np.log(np.clip(p[np.arange(len(c)), c], 1e-10, 1 - 1e-10)) for p, c in zip(probs, choices)]
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
    prog = program.literature(literature.LITERATURE[name][0])
    prog_probs = [np.asarray(program.replay(prog, params, c, r)[0]) for c, r in zip(choices, rewards)]
    diff = max(float(np.max(np.abs(a - b))) for a, b in zip(ref_probs, prog_probs))
    pooled_ref, pooled_prog = pooled_likelihood(ref_probs, choices), pooled_likelihood(prog_probs, choices)
    return {"max_abs_diff": diff, "pooled_reference": pooled_ref, "pooled_program": pooled_prog,
            "pooled_diff": abs(pooled_ref - pooled_prog), "n_trials": int(sum(len(c) for c in choices)),
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

"""Evaluate one program on a set of subjects: fit on fit sessions, score score sessions.

Subjects are independent, so they run in parallel, one single-threaded JAX process each; a
112-core node evaluates about 100 subjects at once. Each worker receives only its subject's
sessions, never the whole freeze.

    regime "subject":  per-subject MLE, point scores
    regime "session":  per-subject MLE as a start, then the session-level MAP, session marginal
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np

from .workers import WORKER_ENV



@dataclass
class SubjectData:
    subject_id: str
    fit: list      # mevolve.data.Session objects (role "fit")
    score: list    # role "score"


def subject_data(freeze, subject_id: str) -> SubjectData:
    return SubjectData(subject_id, freeze.subject_sessions(subject_id, "fit"),
                       freeze.subject_sessions(subject_id, "score"))


def _evaluate_one(args) -> dict:
    program_path, data, regime, seed, n_draws, n_starts, max_em = args
    import jax

    from . import fitting, program, scoring
    from .fingerprints import fingerprints

    t0 = time.time()
    prog = program.load(program_path)
    fit_batch, score_batch = fitting.pad(data.fit), fitting.pad(data.score)
    per = fitting.fit_per_subject(prog, fit_batch, n_starts=n_starts, seed=seed)
    fit_ll, _ = scoring.point_scores(prog, per["x"], fit_batch)
    out = {"subject_id": data.subject_id, "params": per["params"], "fit_nll": per["nll"],
           "fit_log_lik": float(fit_ll.sum()), "n_fit_trials": fit_batch.n_trials}
    if regime == "subject":
        ll, n = scoring.point_scores(prog, per["x"], score_batch)
        p_right = scoring.point_p_right(prog, per["x"], score_batch)
    else:
        sl = fitting.fit_session_level(prog, fit_batch, per["x"], max_em=max_em)
        # common random draws per subject, from a stable hash of its id
        key = jax.random.PRNGKey(seed + zlib.crc32(data.subject_id.encode()) % 100_000)
        m = scoring.marginal_scores(prog, sl["mu"], sl["log_sigma"], score_batch, key, n_draws=n_draws)
        ll, n = m["log_lik"], m["n_trials"]
        p_right = scoring.marginal_p_right(prog, sl["mu"], sl["log_sigma"], score_batch, key, n_draws=n_draws)
        out.update({"typical_params": sl["typical_params"], "log_sigma": np.asarray(sl["log_sigma"]).tolist(),
                    "ess": m["ess"][: len(data.score)].tolist(),
                    "em_iterations": sl["em_iterations"], "converged": bool(sl["converged"])})
    k = len(data.score)
    out["fingerprints"] = fingerprints([s for s in data.score],
                                       [p_right[i, : len(s.choices)] for i, s in enumerate(data.score)])
    out.update({"session_ids": [s.session_id for s in data.score],
                "session_groups": [f"{s.stage_group}|{s.task}" for s in data.score],
                "session_log_lik": ll[:k].tolist(),
                "session_n_trials": n[:k].astype(int).tolist(), "log_lik": float(ll[:k].sum()),
                "n_trials": int(n[:k].sum()), "seconds": time.time() - t0})
    return out


def evaluate(program_path, subjects: list[SubjectData], regime: str = "session", seed: int = 0,
             n_draws: int = 2048, n_starts: int = 8, workers: int | None = None,
             max_em: int = 20) -> list[dict]:
    """Fit and score every subject; returns one record per subject, in input order. ``max_em``
    caps the session-level EM: 20 for the search, more to run it to convergence for reported
    scores (ADR 0004)."""
    jobs = [(str(program_path), d, regime, seed, n_draws, n_starts, max_em) for d in subjects]
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers == 1:
        return [_evaluate_one(j) for j in jobs]
    saved = {k: os.environ.get(k) for k in WORKER_ENV}
    os.environ.update(WORKER_ENV)      # spawned workers read these at start
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(_evaluate_one, jobs, chunksize=1)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

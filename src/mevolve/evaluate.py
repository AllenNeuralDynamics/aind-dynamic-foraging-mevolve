"""The evaluator: the search's use of the fitter (``mevolve.fitter``). For each subject it fits a
program on the fit sessions, scores the score (held-out) sessions, and returns the score and the
feedback (split diagnostics, replay fingerprints).

Subjects are independent, so they run in parallel, one single-threaded JAX process each; a
112-core node evaluates about 100 subjects at once. Each worker receives only its subject's
sessions, never the whole freeze.

    regime "subject":  per-subject MLE, point scores
    regime "2level":   per-subject MLE as a start, then the 2-level model (session -> subject)
                       fitted by Laplace-EM, scored by the session marginal
"""
from __future__ import annotations

import builtins
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path
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

    from . import fitter, fitting, isolation, program, scoring
    from .fingerprints import fingerprints

    if mp.parent_process() is not None:     # a pool worker: no files or network for the program (#40)
        isolation.install(program_path)
    t0 = time.time()
    prog = program.load(program_path)
    f = fitter.fit(prog, data.fit, regime, seed=seed, n_starts=n_starts, max_em=max_em)
    score_batch = fitting.pad(data.score)
    fit_ll, _ = scoring.point_scores(prog, f.x, f.batch)
    out = {"subject_id": data.subject_id, "params": f.params, "fit_nll": f.nll,
           "fit_log_lik": float(fit_ll.sum()), "n_fit_trials": f.batch.n_trials}
    if regime == "subject":
        ll, n = scoring.point_scores(prog, f.x, score_batch)
        p_right = scoring.point_p_right(prog, f.x, score_batch)
    else:
        # common random draws per subject, from a stable hash of its id
        key = jax.random.PRNGKey(seed + zlib.crc32(data.subject_id.encode()) % 100_000)
        m = scoring.marginal_scores(prog, f.mu, f.log_sigma, score_batch, key, n_draws=n_draws)
        ll, n = m["log_lik"], m["n_trials"]
        p_right = scoring.marginal_p_right(prog, f.mu, f.log_sigma, score_batch, key, n_draws=n_draws)
        # the per-subject fit at the same held-out sessions: the 2-level model's spread-0 limit
        ps_ll, _ = scoring.point_scores(prog, f.x, score_batch)
        out.update({"typical_params": f.typical_params, "log_sigma": np.asarray(f.log_sigma).tolist(),
                    "per_subject_log_lik": float(ps_ll[: len(data.score)].sum()),
                    "ess": m["ess"][: len(data.score)].tolist(),
                    "em_iterations": f.em_iterations, "converged": f.converged})
    k = len(data.score)
    out["fingerprints"] = fingerprints([s for s in data.score],
                                       [p_right[i, : len(s.choices)] for i, s in enumerate(data.score)])
    out.update({"session_ids": [s.session_id for s in data.score],
                "session_groups": [f"{s.stage_group}|{s.task}" for s in data.score],
                "session_log_lik": ll[:k].tolist(),
                "session_n_trials": n[:k].astype(int).tolist(), "log_lik": float(ll[:k].sum()),
                "n_trials": int(n[:k].sum()), "seconds": time.time() - t0})
    return out


def _die_with_parent() -> None:
    """Pool initializer: the worker gets SIGTERM when its parent dies (Linux PR_SET_PDEATHSIG), so an
    evaluation killed by the search's time limit leaves no orphaned workers holding the node."""
    import ctypes
    import signal
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # 1 = PR_SET_PDEATHSIG
    except OSError:      # not Linux: nothing to set
        return
    parent = mp.parent_process()
    if parent is not None and os.getppid() != parent.pid:    # the parent died before prctl took effect
        os._exit(1)


def program_score(records: list[dict]) -> dict:
    """A program's pooled held-out likelihood per trial from its subjects' 2-level records.

    The 2-level model contains the per-subject model (spread 0), so a pooled 2-level score below
    the pooled per-subject score means the 2-level fit failed, as it does on stepped or flat
    likelihoods (study 12 r3: Rescorla-Wagner, LossCounting, Grossman-MetaLearning). The program
    then keeps its per-subject score, flagged, rather than being lost to the fitter.
    """
    n = sum(r["n_trials"] for r in records)
    two_level = float(np.exp(sum(r["log_lik"] for r in records) / n))
    per_subject = float(np.exp(sum(r["per_subject_log_lik"] for r in records) / n))
    failed = not np.isfinite(two_level) or two_level < per_subject     # NaN or lower: the fit failed
    return {"score": per_subject if failed else two_level, "two_level": two_level,
            "per_subject": per_subject, "two_level_failed": failed}


def json_call(fn, args) -> str:
    """In a worker: ``fn(*args)`` as JSON, or the error's type name and message. Only plain JSON crosses
    back to the parent, never a pickle a candidate could shape (#40)."""
    try:
        return json.dumps({"ok": fn(*args)}, default=_plain)
    except BaseException as e:      # noqa: BLE001 - every failure crosses as text
        try:
            return json.dumps({"error": [str(type(e).__name__), str(e)]})
        except BaseException:       # noqa: BLE001 - even describing it failed (a candidate's __str__)
            return '{"error": ["RuntimeError", "a candidate raised an error that could not be described"]}'


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"{type(x).__name__} does not cross from a worker")


def from_json(text: str):
    """``json_call``'s result in the parent; a worker's error is raised again as a built-in exception of the
    same name (``ProgramRejected`` for the checks'), else as ``RuntimeError``."""
    out = json.loads(text)
    if "error" not in out:
        return out["ok"]
    name, message = out["error"]
    if name == "ProgramRejected":
        from .search.checks import ProgramRejected
        raise ProgramRejected(message)
    cls = getattr(builtins, name, None)
    raise (cls if isinstance(cls, type) and issubclass(cls, Exception) else RuntimeError)(
        message if cls else f"{name}: {message}")


def evaluate(program_path, subjects: list[SubjectData], regime: str = "2level", seed: int = 0,
             n_draws: int = 2048, n_starts: int = 8, workers: int | None = None,
             max_em: int = 20, isolate: bool = False) -> list[dict]:
    """Fit and score every subject; returns one record per subject, in input order. ``max_em``
    caps Laplace-EM: 20 for the search, more to run it to convergence for reported
    scores (ADR 0004). Pool workers are isolated (#40); ``workers=1`` runs here, unisolated, unless
    ``isolate`` (the search's candidates) asks for a pool of one."""
    jobs = [(str(Path(program_path).resolve()), d, regime, seed, n_draws, n_starts, max_em) for d in subjects]
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers == 1 and not isolate:
        return [_evaluate_one(j) for j in jobs]
    saved = {k: os.environ.get(k) for k in WORKER_ENV}
    os.environ.update(WORKER_ENV)      # spawned workers read these at start
    try:
        with mp.get_context("spawn").Pool(workers, initializer=_die_with_parent) as pool:
            return [from_json(r) for r in pool.starmap(json_call, [(_evaluate_one, (j,)) for j in jobs], chunksize=1)]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

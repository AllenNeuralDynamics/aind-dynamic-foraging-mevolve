"""The evaluation script Shinka runs for each candidate program.

    python shinka_eval.py --program_path <gen_N/main.py> --results_dir <gen_N/results> \\
        --freeze <dir> --cohorts <cohorts.json> --generation <name> --validation <name> \\
        --workers 16 [--max_em 20] [--n_draws 2048]

It writes Shinka's two files: ``correct.json`` (``correct``, ``error``) and ``metrics.json``
(``combined_score``, ``public``, ``private``, ``text_feedback``). ``combined_score`` is the
program's score on the validation subjects (``evaluate.program_score`` of its 2-level records)
x100, which ranks programs; the feedback comes from the generation subjects (study 40 ADR 0003).

Shinka runs this file as a script, so it imports ``mevolve`` absolutely. A candidate is code
Claude wrote. Before any of it runs, the API keys leave this process's
environment and its source must pass ``checks.check_source``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

SECRETS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN")
STATS = (("losses", "after consecutive unrewarded trials on this side"),
         ("pattern", "by the last 1-3 trials' sides and rewards"),
         ("run", "by last reward x current run length"))


def _pooled(records: list[dict]) -> dict[str, tuple[float, int]]:
    """Per (stage group | task): pooled likelihood x100 and scored trials."""
    acc: dict[str, list[float]] = {}
    for r in records:
        for group, ll, n in zip(r["session_groups"], r["session_log_lik"], r["session_n_trials"]):
            a = acc.setdefault(group, [0.0, 0])
            a[0] += ll
            a[1] += n
    return {g: (100 * math.exp(ll / n), n) for g, (ll, n) in sorted(acc.items(), key=lambda kv: -kv[1][1]) if n}


def _params(records: list[dict], bounds: dict[str, tuple[float, float]]) -> list[str]:
    """Median typical value across subjects with its 10th-90th percentile, and a flag when most
    subjects sit at a bound."""
    import numpy as np

    lines = []
    for name, (lo, hi) in bounds.items():
        values = np.array([r["typical_params"][name] for r in records if "typical_params" in r])
        if values.size == 0:
            continue
        q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        tol = 0.01 * (hi - lo)
        at_bound = np.mean((values <= lo + tol) | (values >= hi - tol))
        flag = f"  <- at a bound for {at_bound:.0%} of mice" if at_bound > 0.5 else ""
        lines.append(f"  {name} {q50:.3g} [{q10:.3g} to {q90:.3g}] (bounds {lo:g} to {hi:g}){flag}")
    return lines


def feedback(validation: dict, generation: dict, gen_records: list[dict], bounds: dict) -> str:
    """The text Claude reads about this program."""
    import numpy as np

    lines = [f"Validation score {100 * validation['score']:.3f}"
             + (" (2-level fit FAILED on the validation mice; scored with one parameter set per mouse)"
                if validation["two_level_failed"] else ""),
             f"Generation mice: score {100 * generation['score']:.3f}"
             + (" (2-level fit failed here too)" if generation["two_level_failed"] else ""), "",
             "Generation mice, by stage and task (score, scored trials):"]
    lines += [f"  {g:40s} {s:7.2f}  {n:>8,}" for g, (s, n) in _pooled(gen_records).items()]
    lines += ["", "Generation mice, mean |predicted - observed| switch probability (each mouse counts once):"]
    for key, label in STATS:
        gaps = [r["fingerprints"]["all"][key] for r in gen_records if key in r.get("fingerprints", {}).get("all", {})]
        if gaps:
            lines.append(f"  {label}: {np.mean(gaps):.3f}")
    lines += ["", "Fitted parameters on the generation mice (median typical value [10th to 90th percentile]):"]
    lines += _params(gen_records, bounds)
    return "\n".join(lines)


def evaluate_program(program_path: Path, freeze_dir: Path, cohorts_path: Path, generation: str,
                     validation: str, workers: int, max_em: int = 20, n_draws: int = 2048) -> dict:
    """Fit and score one program on the generation and validation subjects; returns Shinka's
    metrics dict. Raises ``checks.ProgramRejected`` for a program that fails the static checks."""
    from mevolve import evaluate, program
    from mevolve.data import Freeze
    from mevolve.search.checks import check_source

    t0 = time.time()
    shape = check_source(Path(program_path).read_text())
    prog = program.load(program_path)       # executes the candidate: only after the checks
    cohorts = json.loads(Path(cohorts_path).read_text())["cohorts"]
    freeze = Freeze(freeze_dir)
    gen_ids, val_ids = cohorts[generation], cohorts[validation]
    subjects = [evaluate.subject_data(freeze, s) for s in gen_ids + val_ids]
    records = evaluate.evaluate(program_path, subjects, regime="2level", workers=workers, max_em=max_em,
                                n_draws=n_draws)
    gen_records, val_records = records[:len(gen_ids)], records[len(gen_ids):]
    val_score, gen_score = evaluate.program_score(val_records), evaluate.program_score(gen_records)
    seconds = [r["seconds"] for r in records]
    return {
        "combined_score": 100 * val_score["score"],
        "public": {"validation_score": round(100 * val_score["score"], 3),
                   "generation_score": round(100 * gen_score["score"], 3),
                   "two_level_failed": bool(val_score["two_level_failed"]), **shape},
        "private": {"validation": val_score, "generation": gen_score, "seconds": time.time() - t0,
                    "subject_seconds_max": max(seconds), "max_em": max_em, "n_draws": n_draws,
                    "cohorts": [generation, validation], "n_subjects": [len(gen_ids), len(val_ids)]},
        "text_feedback": feedback(val_score, gen_score, gen_records, prog.params),
    }


def write_results(results_dir: Path, metrics: dict | None, error: str | None) -> bool:
    """Shinka's ``correct.json`` and ``metrics.json``; a missing or non-finite score is incorrect."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    score = (metrics or {}).get("combined_score")
    correct = error is None and score is not None and math.isfinite(score)
    if error is None and not correct:
        error = f"non-finite score {score!r}"
    if not correct:
        metrics = {"combined_score": 0.0, "public": {}, "private": {},
                   "text_feedback": f"The program failed and was not scored: {error}"}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float) + "\n")
    (results_dir / "correct.json").write_text(json.dumps({"correct": correct, "error": error}) + "\n")
    return correct


def main(argv: list[str] | None = None) -> int:
    for key in SECRETS:                       # before any candidate code can read them
        os.environ.pop(key, None)
    parser = argparse.ArgumentParser(description="Evaluate one candidate program for Shinka")
    parser.add_argument("--program_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max_em", type=int, default=20)
    parser.add_argument("--n_draws", type=int, default=2048)
    args = parser.parse_args(argv)
    from mevolve.search.checks import ProgramRejected

    metrics, error = None, None
    try:
        metrics = evaluate_program(args.program_path, args.freeze, args.cohorts, args.generation,
                                   args.validation, args.workers, args.max_em, args.n_draws)
    except ProgramRejected as e:
        error = f"rejected by the static checks: {e}"
    except Exception as e:  # noqa: BLE001 - any failure of a candidate is a failed candidate
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    correct = write_results(args.results_dir, metrics, error)
    print(json.dumps({"correct": correct, "error": error,
                      "combined_score": (metrics or {}).get("combined_score")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

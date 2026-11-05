"""The evaluation script Shinka runs for each candidate program.

    python shinka_eval.py --program_path <gen_N/main.py> --results_dir <gen_N/results> \\
        --freeze <dir> --cohorts <cohorts.json> --generation <name> --validation <name> \\
        --workers 16 [--max_em 20] [--n_draws 2048] [--repeat_cache <dir>]

It writes Shinka's two files: ``correct.json`` (``correct``, ``error``) and ``metrics.json``
(``combined_score``, ``public``, ``private``, ``text_feedback``). ``combined_score`` is the
program's score on the validation subjects (``evaluate.program_score`` of its 2-level records)
x100, which ranks programs; the feedback comes from the generation subjects (study 40 ADR 0003).

With ``--repeat_cache``, a program whose code repeats an earlier correct one (the same syntax tree,
comments, formatting and docstrings aside) on the same data and settings reuses that result instead
of being refitted: the evaluator is deterministic, so a refit would give the same metrics.

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
    if generation["two_level_failed"]:      # the details below would describe the rejected 2-level fit
        lines[-1] = ("Details by stage and task, switch probabilities and fitted parameters are omitted: the "
                     "2-level fit failed on the generation mice, and only the per-mouse fit's total score is kept.")
        return "\n".join(lines)
    lines += [f"  {g:40s} {s:7.2f}  {n:>8,}" for g, (s, n) in _pooled(gen_records).items()]
    lines += ["", "Generation mice, mean |predicted - observed| switch probability (each mouse counts once):"]
    for key, label in STATS:
        gaps = [r["fingerprints"]["all"][key] for r in gen_records if key in r.get("fingerprints", {}).get("all", {})]
        if gaps:
            lines.append(f"  {label}: {np.mean(gaps):.3f}")
    lines += ["", "Fitted parameters on the generation mice (median typical value [10th to 90th percentile]):"]
    lines += _params(gen_records, bounds)
    return "\n".join(lines)


def check_shapes(prog) -> None:
    """Trace one step of ``prog`` (no data, no compute) and refuse it unless ``init_latents`` and
    ``update`` return ``N_LATENT`` values and ``predict`` returns 2: the latent limit then holds for
    what is fitted, not only for what is declared."""
    import jax
    import jax.numpy as jnp

    from mevolve.program import SESSION_FACTS
    from mevolve.search.checks import ProgramRejected

    p = {k: jnp.float64((lo + hi) / 2) for k, (lo, hi) in prog.params.items()}
    session = {k: jnp.float64(0.0) for k in SESSION_FACTS}

    def step():
        latents = jnp.asarray(prog.init_latents(p, session), dtype=jnp.float64)
        return (latents, jnp.asarray(prog.predict(latents, p)),
                jnp.asarray(prog.update(latents, jnp.int32(1), jnp.float64(1.0), p)))
    try:
        init, logits, updated = jax.eval_shape(step)
    except Exception as e:
        raise ProgramRejected(f"one step of the program fails: {type(e).__name__}: {e}") from e
    want = (prog.n_latent,)
    for what, got in (("init_latents", init.shape), ("update", updated.shape)):
        if got != want:
            raise ProgramRejected(f"{what} returns shape {got}, but N_LATENT is {prog.n_latent}")
    if logits.shape != (2,):
        raise ProgramRejected(f"predict returns shape {logits.shape}, not (2,) logits over [left, right]")


def evaluate_program(program_path: Path, freeze_dir: Path | None, cohorts_path: Path | None, generation: str | None,
                     validation: str | None, workers: int, max_em: int = 20, n_draws: int = 2048,
                     simulated: tuple[Path, Path] | None = None) -> dict:
    """Fit and score one program on the generation and validation subjects; returns Shinka's
    metrics dict. Raises ``checks.ProgramRejected`` for a program that fails the static checks.
    ``simulated`` (generation file, validation file) replaces the freeze and cohorts with subjects
    saved by ``mevolve.gate4``."""
    from mevolve import evaluate, program
    from mevolve.data import Freeze
    from mevolve.search.checks import check_source

    t0 = time.time()
    shape = check_source(Path(program_path).read_text())
    prog = program.load(program_path)       # executes the candidate: only after the checks
    if (len(prog.params), prog.n_latent) != (shape["n_params"], shape["n_latent"]):
        from mevolve.search.checks import ProgramRejected
        raise ProgramRejected(f"the loaded program has {len(prog.params)} parameters and {prog.n_latent} latent "
                              f"variables, not the {shape['n_params']} and {shape['n_latent']} its literals declare")
    check_shapes(prog)
    if simulated:
        from mevolve.gate4 import load_simulated
        gen_subjects, val_subjects = (load_simulated(p) for p in simulated)
        generation, validation = (Path(p).name for p in simulated)
    else:
        cohorts = json.loads(Path(cohorts_path).read_text())["cohorts"]
        freeze = Freeze(freeze_dir)
        gen_subjects = [evaluate.subject_data(freeze, s) for s in cohorts[generation]]
        val_subjects = [evaluate.subject_data(freeze, s) for s in cohorts[validation]]
    subjects = gen_subjects + val_subjects
    records = evaluate.evaluate(program_path, subjects, regime="2level", workers=workers, max_em=max_em,
                                n_draws=n_draws)
    gen_records, val_records = records[:len(gen_subjects)], records[len(gen_subjects):]
    val_score, gen_score = evaluate.program_score(val_records), evaluate.program_score(gen_records)
    seconds = [r["seconds"] for r in records]
    return {
        "combined_score": 100 * val_score["score"],
        "public": {"validation_score": round(100 * val_score["score"], 3),
                   "generation_score": round(100 * gen_score["score"], 3),
                   "two_level_failed": bool(val_score["two_level_failed"]), **shape},
        "private": {"validation": val_score, "generation": gen_score, "seconds": time.time() - t0,
                    "subject_seconds_max": max(seconds), "max_em": max_em, "n_draws": n_draws,
                    "cohorts": [generation, validation], "n_subjects": [len(gen_subjects), len(val_subjects)]},
        "text_feedback": feedback(val_score, gen_score, gen_records, prog.params),
    }


def code_key(source: str) -> str | None:
    """sha256 of the program's syntax tree without comments, formatting or docstrings (bare string
    statements): equal keys mean the same code. Names, constants and statements all count. None if
    the source does not parse."""
    import ast
    import hashlib

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                                 and isinstance(s.value.value, str))] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def repeat_key(source: str, simulated: tuple[Path, Path] | None, freeze_dir: Path | None, cohorts_path: Path | None,
               generation: str | None, validation: str | None, workers: int, max_em: int, n_draws: int) -> str | None:
    """The repeat cache's key: the code (``code_key``), the data, the fit settings and the evaluator's
    source, as for the seed cache. None if the code does not parse."""
    import hashlib

    from mevolve.search.run import evaluator_digest

    code = code_key(source)
    if code is None:
        return None
    if simulated:
        data = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in simulated]
    else:
        data = [Path(cohorts_path).read_text(), generation, validation, (Path(freeze_dir) / "manifest.json").read_text()]
    return hashlib.sha256(json.dumps([code, *data, workers, max_em, n_draws, evaluator_digest()]).encode()).hexdigest()


def _cached(entry: Path) -> dict | None:
    """A repeat-cache entry, or None: a missing or unreadable entry is a miss, so the program is evaluated."""
    try:
        hit = json.loads(entry.read_text())
        if isinstance(hit["program"], str) and isinstance(hit["metrics"]["private"], dict):
            return hit
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


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
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--generation")
    parser.add_argument("--validation")
    parser.add_argument("--simulated_generation", type=Path, help="mevolve.gate4 subjects, instead of the freeze")
    parser.add_argument("--simulated_validation", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max_em", type=int, default=20)
    parser.add_argument("--n_draws", type=int, default=2048)
    parser.add_argument("--repeat_cache", type=Path, help="this run's earlier correct results, by repeat_key")
    args = parser.parse_args(argv)
    if bool(args.simulated_generation) != bool(args.simulated_validation):
        parser.error("give both --simulated_generation and --simulated_validation, or neither")
    simulated = (args.simulated_generation, args.simulated_validation) if args.simulated_generation else None
    if simulated and any((args.freeze, args.cohorts, args.generation, args.validation)):
        parser.error("the --simulated_ files replace --freeze, --cohorts, --generation and --validation")
    if not simulated and not all((args.freeze, args.cohorts, args.generation, args.validation)):
        parser.error("give --freeze, --cohorts, --generation and --validation, or the two --simulated_ files")
    from mevolve.search.checks import ProgramRejected, check_source

    metrics, error, hit, key = None, None, None, None
    try:
        if args.repeat_cache:
            source = Path(args.program_path).read_text()
            key = repeat_key(source, simulated, args.freeze, args.cohorts, args.generation, args.validation,
                             args.workers, args.max_em, args.n_draws)
            hit = _cached(args.repeat_cache / f"{key}.json") if key else None
        if hit:                 # the same code was scored: its checks again (on this source), then its metrics
            check_source(source)
            metrics = {**hit["metrics"], "private": {**hit["metrics"]["private"], "repeat_of": hit["program"]}}
        else:
            metrics = evaluate_program(args.program_path, args.freeze, args.cohorts, args.generation,
                                       args.validation, args.workers, args.max_em, args.n_draws, simulated)
    except ProgramRejected as e:
        error = f"rejected by the static checks: {e}"
    except Exception as e:  # noqa: BLE001 - any failure of a candidate is a failed candidate
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    correct = write_results(args.results_dir, metrics, error)
    if key and correct and not hit:        # failures are never stored: they are evaluated again
        args.repeat_cache.mkdir(parents=True, exist_ok=True)
        tmp = args.repeat_cache / f".{key}.{os.getpid()}"
        tmp.write_text(json.dumps({"program": str(args.program_path), "metrics": metrics}, default=float) + "\n")
        tmp.replace(args.repeat_cache / f"{key}.json")
    print(json.dumps({"correct": correct, "error": error,
                      "combined_score": (metrics or {}).get("combined_score")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

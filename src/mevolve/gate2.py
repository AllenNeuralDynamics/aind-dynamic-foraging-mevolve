"""Gate 2: one full evaluation (generation + validation cohorts) is fast and deterministic, and the
paired standard error between two programs is measured for the search's stopping rule.

For each program, the generation and validation subjects are fitted and scored at the session
level, one process per subject; the wall time of the whole evaluation is what one proposal costs.
The first program is evaluated twice, and the two scores must be identical. For each pair of
programs, the standard error of their score difference on the validation subjects is computed with
the subjects as the unit (score = 100 x pooled likelihood, as the search sees it).

    python -m mevolve.gate2 --freeze <dir> --cohorts cohorts.json --out gate2.json --workers 112 \\
        --model Hattori2019 --model CompareToThreshold
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import evaluate, literature, program
from .data import Freeze


def paired_se(a: list[dict], b: list[dict]) -> dict:
    """Score difference (x100) between two programs on the same subjects, and its standard error.

    Each score is 100 exp(m) with m = sum LL_i / sum N_i, a ratio estimator. With subjects as the
    unit, the influence of subject i on m is (LL_i - m N_i) / sum N, so a subject's length alone
    adds nothing; the joint delta method over (m_a, m_b), with their covariance, carries both to
    the score difference 100 (exp(m_a) - exp(m_b)).
    """
    la = {r["subject_id"]: r["log_lik"] for r in a}
    n_i = np.array([r["n_trials"] for r in b], dtype=float)
    ll_a = np.array([la[r["subject_id"]] for r in b])
    ll_b = np.array([r["log_lik"] for r in b])
    n_tot = n_i.sum()
    m_a, m_b = ll_a.sum() / n_tot, ll_b.sum() / n_tot
    infl = np.stack([ll_a - m_a * n_i, ll_b - m_b * n_i]) / n_tot          # (2, n_subjects)
    cov = len(n_i) * np.cov(infl, ddof=1)
    grad = 100 * np.array([np.exp(m_a), -np.exp(m_b)])
    diff = 100 * (np.exp(m_a) - np.exp(m_b))
    se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    return {"score_diff_x100": float(diff), "se_x100": float(se), "n_subjects": len(n_i)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate 2: evaluation time, determinism, paired SE")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True, help="cohort lists (mevolve.cohorts output)")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    freeze = Freeze(args.freeze)
    cohorts = json.loads(args.cohorts.read_text())["cohorts"]
    gen, val = cohorts["generation"], cohorts["validation"]
    data = [evaluate.subject_data(freeze, s) for s in gen + val]
    results, runs = {}, {}
    for i, model in enumerate(args.model):
        path = program.LITERATURE_DIR / f"{literature.LITERATURE[model][0]}.py"
        repeats = 2 if i == 0 else 1
        for rep in range(repeats):
            t0 = time.time()
            records = evaluate.evaluate(path, data, regime="session", workers=args.workers)
            wall = time.time() - t0
            v = [r for r in records if r["subject_id"] in set(val)]
            score = 100 * np.exp(sum(r["log_lik"] for r in v) / sum(r["n_trials"] for r in v))
            runs.setdefault(model, []).append({"wall_seconds": wall, "validation_score_x100": float(score),
                                               "max_subject_seconds": max(r["seconds"] for r in records)})
            results[model] = v
            print(f"{model} run {rep + 1}: validation score {score:.6f}, wall {wall:.0f} s", flush=True)
    first = runs[args.model[0]]
    report = {"freeze_manifest_sha256": hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest(),
              "cohorts_sha256": hashlib.sha256(args.cohorts.read_bytes()).hexdigest(),
              "n_generation": len(gen), "n_validation": len(val), "runs": runs,
              "deterministic": first[0]["validation_score_x100"] == first[1]["validation_score_x100"],
              "paired": {f"{a} - {b}": paired_se(results[a], results[b])
                         for a, b in itertools.combinations(args.model, 2)}}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("deterministic", "paired")}, indent=1))


if __name__ == "__main__":
    main()

"""Gate 1a on real sessions: every literature program against its reference forager.

Each subject's sessions (all of them, ignored trials removed) are replayed by both, at one
parameter set per subject: random draws within the reference fit's bounds (``--params random``),
or study 12's fitted parameters (``--params study12``, once its per-subject fits are frozen).
One process per subject; writes one JSON with every subject's result per model.

    python -m mevolve.gate1a --freeze <dir> --out gate1a.json --pool heldout --workers 64 \\
        [--params study12 --studies <studies checkout>] [--model NAME ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .workers import WORKER_ENV

STUDY12_FITS = "studies/12-cognitive-literature-leaderboard/analysis/per_subject_fits.csv"


def _one(args):
    from . import parity

    name, subject_id, choices, rewards, params = args
    r = parity.compare(name, params, choices, rewards)
    return {"subject_id": subject_id, "params": params, **r}


def study12_params(csv_path: Path, name: str) -> dict[str, dict[str, float]]:
    with csv_path.open(newline="") as handle:
        return {row["subject_id"]: json.loads(row["fitted_params"])
                for row in csv.DictReader(handle) if row["literature_model"] == name}


def main(argv: list[str] | None = None) -> None:
    from . import literature, parity
    from .data import Freeze

    parser = argparse.ArgumentParser(description="Gate 1a on real sessions")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", default="heldout")
    parser.add_argument("--params", choices=["random", "study12"], default="random")
    parser.add_argument("--studies", type=Path)
    parser.add_argument("--study12-csv", type=Path, help="a provisional copy of study 12's per_subject_fits.csv")
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    if args.params == "study12" and not (args.study12_csv or args.studies):
        parser.error("--params study12 needs --studies or --study12-csv")
    freeze = Freeze(args.freeze)
    subjects = freeze.subjects(args.pool)
    sessions = {s: freeze.subject_sessions(s) for s in subjects}
    models = args.model or list(literature.LITERATURE)
    os.environ.update(WORKER_ENV)
    report = {"pool": args.pool, "params": args.params, "seed": args.seed,
              "freeze_manifest_sha256": hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest(),
              "tolerances": {"per_trial": parity.PER_TRIAL_TOL, "pooled": parity.POOLED_TOL}, "models": {}}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name in models:
            t0 = time.time()
            if args.params == "study12":
                fitted = study12_params(args.study12_csv or args.studies / STUDY12_FITS, name)
                _, clamped = literature.fit_spec(name)
                per_subject = {s: {k: v for k, v in fitted[s].items() if k not in clamped} for s in subjects}
            else:
                # seeded by the model's place in the whole library, so a rerun of one model draws the same
                rng = np.random.default_rng([args.seed, list(literature.LITERATURE).index(name)])
                per_subject = {s: parity.draw_params(name, rng) for s in subjects}
            jobs = [(name, s, [x.choices for x in sessions[s]], [x.rewards for x in sessions[s]], per_subject[s])
                    for s in subjects]
            rows = pool.map(_one, jobs, chunksize=1)
            summary = {"ok": all(r["ok"] for r in rows), "n_subjects": len(rows),
                       "n_trials": int(sum(r["n_trials"] for r in rows)),
                       "n_excluded": int(sum(r["n_excluded"] for r in rows)),
                       "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                       "max_pooled_diff": max(r["pooled_diff"] for r in rows),
                       "failed_subjects": [r["subject_id"] for r in rows if not r["ok"]],
                       "seconds": time.time() - t0}
            report["models"][name] = {"summary": summary, "subjects": rows}
            print(f"{name}: {'PASS' if summary['ok'] else 'FAIL'} max diff {summary['max_abs_diff']:.2e}, "
                  f"pooled {summary['max_pooled_diff']:.2e}, excluded {summary['n_excluded']}, "
                  f"{summary['seconds']:.0f} s", flush=True)
    report["ok"] = all(m["summary"]["ok"] for m in report["models"].values())
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

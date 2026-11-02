"""Scores of the literature models as 2-level models on any freeze pool, as reported (ADR 0004).

The 2-level model (session -> subject) is fitted by Laplace-EM run to convergence (``--max-em``,
default 100), and the session marginal uses 2,048 draws; the per-subject fit that starts it is the fitter's default. Models are
named as study 12 names them, so each record joins to its per-subject fits. Writes one JSON per
model, with gate1's per-subject records plus each subject's pool, and refuses a checkout with
uncommitted changes, since the MEvolve SHA is the record's provenance:

    python -m mevolve.score --model Hattori2019 --pool all --freeze <dir> --out <dir> --workers 112
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import evaluate, literature, program
from .data import Freeze
from .freeze import _git_sha

REPORTED_MAX_EM = 100


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Literature models as 2-level models, as reported")
    parser.add_argument("--model", action="append", choices=list(literature.LITERATURE),
                        help="a literature model, as study 12 names it (repeatable; default all)")
    parser.add_argument("--pool", choices=["train", "heldout", "all"], default="all")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-em", type=int, default=REPORTED_MAX_EM)
    parser.add_argument("--n-draws", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_em < 1 or args.n_draws < 1:
        parser.error("--max-em and --n-draws must be at least 1")
    sha = _git_sha(Path(__file__).resolve().parents[2])
    freeze = Freeze(args.freeze)
    pools = list(freeze.manifest["pools"]) if args.pool == "all" else [args.pool]
    subjects = [(p, s) for p in pools for s in freeze.subjects(p)]
    data = [evaluate.subject_data(freeze, s) for _, s in subjects]
    args.out.mkdir(parents=True, exist_ok=True)
    for model in args.model or list(literature.LITERATURE):
        path = program.LITERATURE_DIR / f"{literature.LITERATURE[model][0]}.py"
        t0 = time.time()
        records = evaluate.evaluate(path, data, regime="2level", seed=args.seed, n_draws=args.n_draws,
                                    workers=args.workers, max_em=args.max_em)
        for (pool, _), r in zip(subjects, records, strict=True):
            r["pool"] = pool
        result = {"model": model, "regime": "2level", "pools": pools, "max_em": args.max_em,
                  "n_draws": args.n_draws, "seed": args.seed, "wall_seconds": time.time() - t0,
                  "mevolve_git_sha": sha,
                  "program_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "freeze_manifest_sha256": hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest(),
                  "n_not_converged": sum(not r["converged"] for r in records),
                  "subjects": records}
        (args.out / f"2level_{model}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"{model}: {len(records)} subjects, {result['n_not_converged']} not converged within "
              f"{args.max_em} EM iterations, wall {result['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

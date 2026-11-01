"""Gates 1b and 1c on the 149 test subjects: MEvolve's evaluator against the lab's own fits.

    regime "subject" (1b): per-subject MLE against study 01 r8's per-subject fits of Hattori2019,
        Bari2019 and CompareToThreshold (``rl_baseline.json``); study 12's fits of every
        literature model are the full reference once they are frozen.
    regime "session" (1c): session-level MAP and session marginal against study 13's HB-CTT and
        HB-Hattori (``ctt_final_runs.json``, ``hattori_final_runs.json``, ``r5_ctt_pooled.json``).

Writes one JSON per model and regime with every subject's record and the comparison:

    python -m mevolve.gate1 --regime session --model CompareToThreshold --model Hattori2019 \\
        --freeze <dir> --studies <studies checkout> --out <dir> --workers 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import evaluate, literature, program
from .data import Freeze

RL = "studies/01-gru-scaling-law/analysis/rl_baseline.json"
HB_POOLED = "studies/13-hb-subject-vs-gru-heldout/analysis/r5_ctt_pooled.json"
HB_RUNS = {"CompareToThreshold": "studies/13-hb-subject-vs-gru-heldout/analysis/ctt_final_runs.json",
           "Hattori2019": "studies/13-hb-subject-vs-gru-heldout/analysis/hattori_final_runs.json"}
RL_KEY = {"Hattori2019": "hattori", "Bari2019": "bari", "CompareToThreshold": "ctt"}
HB_KEY = {"CompareToThreshold": "hb_ctt", "Hattori2019": "hb_hattori"}


def reference(model: str, regime: str, studies: Path) -> tuple[dict, float, list[str]]:
    """Per-subject reference likelihoods, the pooled reference, and the source files."""
    if regime == "subject":
        rl = json.loads((studies / RL).read_text())["models"][RL_KEY[model]]
        return rl["per_subject_likelihood"], rl["pooled_likelihood_trial_weighted"], [RL]
    rows = json.loads((studies / HB_RUNS[model]).read_text())["rows"]
    per = {r["subject_id"]: float(np.exp(r["heldout/eval_log_lik_sum"] / r["heldout/num_eval_trials"])) for r in rows}
    pooled = json.loads((studies / HB_POOLED).read_text())[HB_KEY[model]]
    return per, pooled, [HB_RUNS[model], HB_POOLED]


def compare(records: list[dict], ref_per: dict, ref_pooled: float) -> dict:
    ours = {r["subject_id"]: float(np.exp(r["log_lik"] / r["n_trials"])) for r in records}
    pooled = float(np.exp(sum(r["log_lik"] for r in records) / sum(r["n_trials"] for r in records)))
    diff = np.array([ours[s] - ref_per[s] for s in ours if s in ref_per])
    return {"pooled": pooled, "pooled_reference": ref_pooled, "pooled_diff": pooled - ref_pooled,
            "n_subjects": len(ours), "per_subject_diff": {"median": float(np.median(diff)),
                                                          "min": float(diff.min()), "max": float(diff.max())},
            "subjects_worse_than_reference_by_0.005": sorted(s for s in ours if s in ref_per and ours[s] < ref_per[s] - 0.005)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gates 1b and 1c on the test subjects")
    parser.add_argument("--regime", choices=["subject", "session"], required=True)
    parser.add_argument("--model", action="append", required=True, help="study 12 name, repeatable")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-draws", type=int, default=256)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = Freeze(args.freeze)
    subjects = [evaluate.subject_data(freeze, s) for s in freeze.subjects("heldout")]
    for model in args.model:
        path = program.LITERATURE_DIR / f"{literature.LITERATURE[model][0]}.py"
        t0 = time.time()
        records = evaluate.evaluate(path, subjects, regime=args.regime, n_draws=args.n_draws, workers=args.workers)
        ref_per, ref_pooled, sources = reference(model, args.regime, args.studies)
        result = {"model": model, "regime": args.regime, "wall_seconds": time.time() - t0,
                  "program_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "freeze_manifest_sha256": hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest(),
                  "sources": [{"path": p, "sha256": hashlib.sha256((args.studies / p).read_bytes()).hexdigest()} for p in sources],
                  "comparison": compare(records, ref_per, ref_pooled), "subjects": records}
        (args.out / f"{args.regime}_{program.load(path).name}.json").write_text(json.dumps(result, indent=1) + "\n")
        c = result["comparison"]
        print(f"{model} {args.regime}: pooled {c['pooled']:.5f} vs {c['pooled_reference']:.5f} "
              f"(diff {c['pooled_diff']:+.5f}), wall {result['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

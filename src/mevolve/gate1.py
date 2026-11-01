"""Gates 1b and 1c on the 149 test subjects: MEvolve's evaluator against the lab's own fits.

    regime "subject" (1b): per-subject MLE against study 12's per-subject differential-evolution
        fits of every literature model (``--reference study12``: its frozen
        ``per_subject_fits.csv``, or ``--study12-csv`` for a provisional copy), or against study 01
        r8's fits of Hattori2019, Bari2019 and CompareToThreshold (``--reference study01``).
        Pass (ADR 0004 of study 40): pooled held-out likelihood within 0.0002 of the reference or
        better, and for at least 95% of subjects a fit-session log likelihood no worse than the
        reference's by more than 0.5 nats.
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
STUDY12 = "studies/12-cognitive-literature-leaderboard/analysis/per_subject_fits.csv"
RL_KEY = {"Hattori2019": "hattori", "Bari2019": "bari", "CompareToThreshold": "ctt"}
POOLED_TOL_1B, FIT_NATS_1B, FIT_SHARE_1B = 0.0002, 0.5, 0.95


def study12_reference(csv_path: Path, model: str) -> dict:
    """Study 12's held-out subjects for one model: eval and fit log likelihoods and trial counts."""
    import csv
    with csv_path.open(newline="") as h:
        rows = [r for r in csv.DictReader(h) if r["literature_model"] == model and r["cohort"] == "heldout"]
    return {r["subject_id"]: {"eval_ll": float(r["eval_total_log_likelihood"]), "eval_n": int(float(r["eval_total_trials"])),
                              "fit_ll": float(r["train_total_log_likelihood"])} for r in rows}


def compare_study12(records: list[dict], ref: dict) -> dict:
    subs = [r for r in records if r["subject_id"] in ref]
    if not subs:
        return {"n_subjects": 0, "subjects_missing_from_reference": len(records), "ok": False}
    pooled = float(np.exp(sum(r["log_lik"] for r in subs) / sum(r["n_trials"] for r in subs)))
    pooled_ref = float(np.exp(sum(ref[r["subject_id"]]["eval_ll"] for r in subs) / sum(ref[r["subject_id"]]["eval_n"] for r in subs)))
    fit_gap = np.array([r["fit_log_lik"] - ref[r["subject_id"]]["fit_ll"] for r in subs])
    trials_differ = [r["subject_id"] for r in subs if r["n_trials"] != ref[r["subject_id"]]["eval_n"]]
    share = float(np.mean(fit_gap >= -FIT_NATS_1B))
    return {"pooled": pooled, "pooled_reference": pooled_ref, "pooled_diff": pooled - pooled_ref,
            "n_subjects": len(subs), "subjects_missing_from_reference": len(records) - len(subs),
            "subjects_with_other_trial_counts": trials_differ,
            "fit_gap_nats": {"median": float(np.median(fit_gap)), "min": float(fit_gap.min())},
            "share_fit_within_0.5_nats": share,
            "ok": (pooled - pooled_ref >= -POOLED_TOL_1B and share >= FIT_SHARE_1B and not trials_differ
                   and len(subs) == len(records))}
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
    parser.add_argument("--model", action="append", required=True,
                        help="study 12 name, repeatable; NAME=PATH scores another program against NAME's reference")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reference", choices=["study12", "study01"], default="study12",
                        help="per-subject regime only: which fits gate 1b compares with")
    parser.add_argument("--study12-csv", type=Path, help="a provisional copy of study 12's per_subject_fits.csv")
    parser.add_argument("--n-draws", type=int, default=2048)
    args = parser.parse_args(argv)
    args.studies = args.studies.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = Freeze(args.freeze)
    if args.regime == "subject" and args.reference == "study12":   # check coverage before any fit
        csv_path = (args.study12_csv or args.studies / STUDY12).resolve()
        for item in args.model:
            ref = study12_reference(csv_path, item.partition("=")[0])
            missing = set(freeze.subjects("heldout")) - set(ref)
            if missing:
                parser.error(f"{item}: {len(missing)} held-out subjects have no row in {csv_path}")
    subjects = [evaluate.subject_data(freeze, s) for s in freeze.subjects("heldout")]
    for item in args.model:
        model, _, override = item.partition("=")
        path = Path(override) if override else program.LITERATURE_DIR / f"{literature.LITERATURE[model][0]}.py"
        t0 = time.time()
        records = evaluate.evaluate(path, subjects, regime=args.regime, n_draws=args.n_draws, workers=args.workers)
        if args.regime == "subject" and args.reference == "study12":
            csv_path = (args.study12_csv or args.studies / STUDY12).resolve()
            comparison = compare_study12(records, study12_reference(csv_path, model))
            sources = [str(csv_path)]
            source_root = Path("/")
        else:
            ref_per, ref_pooled, sources = reference(model, args.regime, args.studies)
            comparison = compare(records, ref_per, ref_pooled)
            source_root = args.studies
        result = {"model": model, "regime": args.regime, "wall_seconds": time.time() - t0,
                  "program_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "freeze_manifest_sha256": hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest(),
                  "sources": [{"path": p, "sha256": hashlib.sha256((source_root / p).read_bytes()).hexdigest()} for p in sources],
                  "comparison": comparison, "subjects": records}
        (args.out / f"{args.regime}_{program.load(path).name}.json").write_text(json.dumps(result, indent=1) + "\n")
        c = result["comparison"]
        extra = f", fit within 0.5 nats {c['share_fit_within_0.5_nats']:.3f}, {'PASS' if c['ok'] else 'FAIL'}" if "ok" in c else ""
        print(f"{model} {args.regime}: pooled {c['pooled']:.5f} vs {c['pooled_reference']:.5f} "
              f"(diff {c['pooled_diff']:+.5f}){extra}, wall {result['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

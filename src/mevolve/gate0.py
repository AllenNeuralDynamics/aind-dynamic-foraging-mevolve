"""Gate 0: the freeze matches the committed subject lists and scored-trial counts.

Every expected value is read from committed JSON in the studies repo, never typed here:

- the held-out subjects are the 149 of the per-subject MLE baselines (study 01 r8);
- their scored sessions and trials are the ones HB-CTT was scored on (study 13 r5), and the MLE
  baselines' scored trials (study 01 r8);
- the training pool and the GRU's split (study 01, copied in study 10) differ only by subjects
  swapped between the two lists.

    python -m mevolve.gate0 --freeze <dir> --studies ~/code/aind-dynamic-foraging-bfm-studies
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RL = "studies/01-gru-scaling-law/analysis/rl_baseline.json"
HB = "studies/13-hb-subject-vs-gru-heldout/analysis/r5_ctt_pooled.json"
GRU_SPLIT = "studies/10-gru-reservoir-ablation/reference/study01-v2-exact-split.json"


def gru_extra_test_subjects(manifest: dict, studies: Path) -> list[str]:
    """Subjects on the GRU's held-out list that sit in our training pool (ADR 0003 keeps them out
    of every search cohort)."""
    gru = json.loads((studies / GRU_SPLIT).read_text())
    train = set(manifest["pools"]["train"]["subjects"])
    return sorted(str(s) for s in gru["test_subject_ids"] if str(s) in train)


def check(manifest: dict, studies: Path) -> dict:
    """Gate 0's checks, each with its value, its expected value and whether it passed."""
    rl = json.loads((studies / RL).read_text())["models"]["ctt"]
    hb = json.loads((studies / HB).read_text())
    gru = json.loads((studies / GRU_SPLIT).read_text())
    held, train = manifest["pools"]["heldout"], manifest["pools"]["train"]
    held_set, train_set = set(held["subjects"]), set(train["subjects"])
    gru_test, gru_train = {str(s) for s in gru["test_subject_ids"]}, {str(s) for s in gru["source_subject_ids"]}

    checks = [
        ("held-out subjects equal the MLE baselines' (study 01 r8)",
         sorted(held_set), sorted(rl["per_subject_likelihood"])),
        ("held-out scored sessions equal HB-CTT's (study 13 r5)",
         held["n_sessions"]["score"], hb["n_sessions"]),
        ("held-out scored trials equal HB-CTT's (study 13 r5)",
         held["n_choice_trials"]["score"], hb["n_trials"]),
        ("held-out scored trials equal the MLE baselines' (study 01 r8)",
         held["n_choice_trials"]["score"], rl["n_trials"]),
        ("training pool has as many subjects as the GRU's", len(train_set), len(gru_train)),
        ("GRU test subjects missing from ours sit in our training pool",
         sorted(gru_test - held_set), sorted(train_set - gru_train)),
        ("our held-out subjects missing from the GRU's test list sit in its training pool",
         sorted(held_set - gru_test), sorted(gru_train - train_set)),
    ]
    rows = [{"check": name, "value": value, "expected": expected, "ok": value == expected}
            for name, value, expected in checks]
    for row in rows:   # keep the record readable: long id lists become counts when they match
        if row["ok"] and isinstance(row["value"], list) and len(row["value"]) > 10:
            row["value"] = row["expected"] = f"{len(row['value'])} subjects, identical"
    sources = [{"path": p, "sha256": hashlib.sha256((studies / p).read_bytes()).hexdigest()}
               for p in (RL, HB, GRU_SPLIT)]
    return {"ok": all(r["ok"] for r in rows), "checks": rows, "sources": sources}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate 0: the freeze matches committed lists and counts")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True, help="studies repo checkout")
    args = parser.parse_args(argv)
    from .data import verify_files

    manifest = json.loads((args.freeze / "manifest.json").read_text())
    verify_files(args.freeze, manifest)
    result = check(manifest, args.studies)
    result["freeze_manifest_sha256"] = hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest()
    (args.freeze / "gate0.json").write_text(json.dumps(result, indent=1) + "\n")
    for row in result["checks"]:
        print(("PASS " if row["ok"] else "FAIL ") + row["check"] + f": {row['value']} vs {row['expected']}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()

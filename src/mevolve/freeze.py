"""Freeze the data and the split once, through the wrapper's own loader (build step 1).

Runs in the shared ``dynamic-foraging-bfm-cpu`` env, which has the wrapper's dependencies and the
database package, with the wrapper checkout's ``code/`` directory on ``sys.path``::

    python -m mevolve.freeze --wrapper ~/code/wt/mevolve-wrapper-331d373 \\
        --out /allen/aind/scratch/han.hou/mevolve/freeze/20260603

The split comes from ``resolve_mice_snapshot_session_split_manifest``, the wrapper's rebuild of
the trainer's split, so the fit and score sessions are the ones every comparator used. Trials come
from ``load_mice_from_database`` with the same settings. Every trial is kept, ignored ones as
response 2, so that the evaluator, not the freeze, applies the ignore policy.

Writes ``trials.parquet``, ``sessions.parquet`` and ``manifest.json`` to ``--out``. Nothing here
belongs in the repo: the freeze lives on scratch and study 40 records its digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

# The dispatcher's data config `mice_snapshot_scaling` at 6450af3, the one study 12 ran.
DATA = {
    "snapshot": "20260603",
    "min_sessions": 10,
    "heldout_every_n": 5,
    "eval_every_n": 2,
    "mature_only": False,
    "ignore_policy": "exclude",
    "curricula": ["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"],
    "subject_ratio": {"Coupled Baiting": 1.0, "Uncoupled Baiting": 1.0, "Uncoupled Without Baiting": 1.0},
    "subject_sample_seed": 0,
}
TRIAL_COLS = ["trial", "subject_id", "ses_idx", "animal_response", "earned_reward", "curriculum_name",
              "task", "current_stage_actual"]
POOLS = {"train": "train", "heldout": "heldout"}   # loader split name -> pool name in the freeze


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(repo: Path) -> str:
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{repo} has uncommitted changes; freeze from a clean checkout")
    return sha


def _subject_curricula() -> dict[str, str]:
    """Each subject's curriculum, assigned by the wrapper's own selection pipeline."""
    from aind_dynamic_foraging_database import select_sessions
    from utils.load_mice_database import _SESSION_METADATA_COLS, _partition_subjects

    sessions = select_sessions(subjects=None, columns=_SESSION_METADATA_COLS, snapshot=DATA["snapshot"])
    sessions["subject_id"] = sessions["subject_id"].astype(str)
    partition = _partition_subjects(sessions, curricula=DATA["curricula"], min_sessions=DATA["min_sessions"],
                                    heldout_every_n=DATA["heldout_every_n"],
                                    subject_ratio=DATA["subject_ratio"], seed=DATA["subject_sample_seed"])
    return {str(s): c for s, c in partition["subject_curriculum"].items()}


def freeze_pool(split: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The wrapper's split manifest, every trial, and one row per session for one pool."""
    from data_loaders.mice import resolve_mice_snapshot_session_split_manifest
    from utils.load_mice_database import load_mice_from_database

    kwargs = {k: DATA[k] for k in ("snapshot", "min_sessions", "heldout_every_n", "mature_only",
                                   "curricula", "subject_ratio")}
    manifest = resolve_mice_snapshot_session_split_manifest(
        split=split, subject_sample_seed=DATA["subject_sample_seed"], ignore_policy=DATA["ignore_policy"],
        eval_every_n=DATA["eval_every_n"], multisubject=True, **kwargs)
    df, _ = load_mice_from_database(split=split, seed=DATA["subject_sample_seed"],
                                    cols_to_retain=TRIAL_COLS, **kwargs)
    pool = POOLS[split]
    rows, keep = [], []
    for subject in manifest["per_subject"]:
        sid = str(subject["subject_id"])
        fit, score = set(subject["train_session_ids"]), set(subject["eval_session_ids"])
        for index, ses in enumerate(subject["full_session_ids"]):
            rows.append({"subject_id": sid, "session_id": ses, "pool": pool, "index": index,
                         "role": "fit" if ses in fit else "score" if ses in score else "none"})
            keep.append(ses)
    sessions = pd.DataFrame(rows)
    trials = df[df["ses_idx"].isin(set(keep))].copy()
    missing = set(keep) - set(trials["ses_idx"])
    if missing:
        raise SystemExit(f"{split}: {len(missing)} manifest sessions have no trials, e.g. {sorted(missing)[:3]}")
    facts = (trials.groupby("ses_idx")
             .agg(task=("task", "first"), stage=("current_stage_actual", "first"),
                  n_trials=("trial", "size"),
                  n_choice=("animal_response", lambda r: int(r.isin([0, 1]).sum())))
             .reset_index().rename(columns={"ses_idx": "session_id"}))
    sessions = sessions.merge(facts, on="session_id", how="left", validate="one_to_one")
    trials = pd.DataFrame({
        "subject_id": trials["subject_id"].astype(str).to_numpy(),
        "session_id": trials["ses_idx"].astype(str).to_numpy(),
        "trial": trials["trial"].astype("int32").to_numpy(),
        "response": trials["animal_response"].astype("int8").to_numpy(),
        "reward": trials["earned_reward"].astype(bool).to_numpy(),
    }).sort_values(["subject_id", "session_id", "trial"], kind="stable")
    return manifest, sessions, trials


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrapper", type=Path, required=True, help="wrapper checkout (clean)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    wrapper_sha = _git_sha(args.wrapper)
    mevolve_sha = _git_sha(Path(__file__).resolve().parents[2])
    sys.path.insert(0, str(args.wrapper / "code"))
    args.out.mkdir(parents=True, exist_ok=False)

    curricula = _subject_curricula()
    manifests, sessions, trials = {}, [], []
    for split in POOLS:
        manifest, ses, tri = freeze_pool(split)
        manifests[POOLS[split]] = manifest
        sessions.append(ses)
        trials.append(tri)
    sessions = pd.concat(sessions, ignore_index=True)
    sessions["curriculum"] = sessions["subject_id"].map(curricula)
    if sessions["curriculum"].isna().any():
        raise SystemExit("a selected subject has no curriculum")
    trials = pd.concat(trials, ignore_index=True)
    sessions.to_parquet(args.out / "sessions.parquet", index=False)
    trials.to_parquet(args.out / "trials.parquet", index=False)

    pools = {}
    for pool, manifest in manifests.items():
        ses = sessions[sessions["pool"] == pool]
        pools[pool] = {
            "selected_subject_ids": [str(s) for s in manifest["selected_subject_ids"]],
            "subjects": [str(r["subject_id"]) for r in manifest["per_subject"]],
            "n_sessions": {role: int((ses["role"] == role).sum()) for role in ("fit", "score")},
            "n_choice_trials": {role: int(ses.loc[ses["role"] == role, "n_choice"].sum())
                                for role in ("fit", "score")},
        }
    record = {
        "data": DATA,
        "wrapper_git_sha": wrapper_sha,
        "mevolve_git_sha": mevolve_sha,
        "packages": {p: metadata.version(p) for p in ("aind-dynamic-foraging-database", "pandas", "pyarrow")},
        "split_source": "data_loaders.mice.resolve_mice_snapshot_session_split_manifest (multisubject)",
        "pools": pools,
        "files": {name: sha256(args.out / name) for name in ("trials.parquet", "sessions.parquet")},
    }
    (args.out / "manifest.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({p: {k: v for k, v in r.items() if k.startswith("n_")} for p, r in pools.items()}))


if __name__ == "__main__":
    main()

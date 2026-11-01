"""A tiny fake freeze and studies checkout, so the data path is tested without /allen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mevolve.gate0 import GRU_SPLIT, HB, RL


def _write_freeze(root: Path) -> Path:
    """Two held-out subjects and three training subjects, four sessions each."""
    sessions, trials = [], []
    pools = {"heldout": ["h1", "h2"], "train": ["t1", "t2", "t3"]}
    for pool, subjects in pools.items():
        for s in subjects:
            for i in range(4):
                sid = f"{s}_2026-01-0{i + 1}"
                responses = [0, 1, 2, 1, 0]      # one ignored trial per session
                sessions.append({"subject_id": s, "session_id": sid, "pool": pool, "index": i,
                                 "role": "fit" if i % 2 == 0 else "score", "task": "Coupled Baiting",
                                 "stage": None if i == 0 else "STAGE_FINAL", "n_trials": 5, "n_choice": 4,
                                 "curriculum": "Coupled Baiting"})
                for t, r in enumerate(responses):
                    trials.append({"subject_id": s, "session_id": sid, "trial": t, "response": r,
                                   "reward": bool(t % 2)})
    root.mkdir()
    pd.DataFrame(sessions).to_parquet(root / "sessions.parquet", index=False)
    pd.DataFrame(trials).astype({"trial": "int32", "response": "int8"}).to_parquet(root / "trials.parquet", index=False)
    ses = pd.DataFrame(sessions)
    manifest = {"pools": {p: {"subjects": subs,
                              "n_sessions": {r: int(((ses.pool == p) & (ses.role == r)).sum()) for r in ("fit", "score")},
                              "n_choice_trials": {r: int(ses.loc[(ses.pool == p) & (ses.role == r), "n_choice"].sum())
                                                  for r in ("fit", "score")}}
                          for p, subs in pools.items()},
                "files": {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                          for n in ("trials.parquet", "sessions.parquet")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def freeze_dir(tmp_path: Path) -> Path:
    return _write_freeze(tmp_path / "freeze")


@pytest.fixture
def studies_dir(tmp_path: Path) -> Path:
    """Committed-JSON stand-ins: held-out h1, h2 (4 scored sessions, 16 scored trials); the GRU's
    lists swap h2 and t3."""
    root = tmp_path / "studies"
    for rel, obj in [
        (RL, {"models": {"ctt": {"per_subject_likelihood": {"h1": 0.7, "h2": 0.7}, "n_trials": 16}}}),
        (HB, {"n_sessions": 4, "n_trials": 16}),
        (GRU_SPLIT, {"test_subject_ids": ["h1", "t3"], "source_subject_ids": ["t1", "t2", "h2"]}),
    ]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(obj))
    return root

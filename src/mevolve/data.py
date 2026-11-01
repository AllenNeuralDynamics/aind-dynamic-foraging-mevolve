"""Read a freeze (build step 1): every session of every subject, with its role and facts.

The evaluator applies the ignore policy here, as the wrapper does: ignored trials (response 2)
are removed before anything sees the session, so a program never receives them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_STAGES = ("STAGE_FINAL", "GRADUATED")


@dataclass(frozen=True)
class Session:
    subject_id: str
    session_id: str
    index: int            # position in the subject's sessions, in date order, from 0
    role: str             # "fit" (even index) or "score" (odd index)
    task: str
    stage: str | None     # current_stage_actual; None when the database has none
    choices: np.ndarray   # 0 = left, 1 = right; ignored trials removed
    rewards: np.ndarray   # 0.0 or 1.0, aligned with choices

    @property
    def stage_group(self) -> str:
        """Final (final or graduated), earlier, or unknown (study 40's stage group)."""
        if self.stage is None:
            return "unknown"
        return "final" if self.stage in FINAL_STAGES else "earlier"


class Freeze:
    """A freeze directory: ``manifest.json``, ``sessions.parquet``, ``trials.parquet``."""

    def __init__(self, path: Path | str, verify: bool = True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if verify:
            for name, digest in self.manifest["files"].items():
                actual = hashlib.sha256((self.path / name).read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f"{name}: sha256 {actual} differs from the manifest's {digest}")
        self.sessions = pd.read_parquet(self.path / "sessions.parquet")
        trials = pd.read_parquet(self.path / "trials.parquet")
        trials = trials[trials["response"] != 2]
        self._trials = {sid: (g["response"].to_numpy(np.int64), g["reward"].to_numpy(np.float64))
                        for sid, g in trials.groupby("session_id", sort=False)}

    def subjects(self, pool: str | None = None) -> list[str]:
        """Subject ids, in the wrapper's order, for one pool (``train``, ``heldout``) or all."""
        pools = [pool] if pool else list(self.manifest["pools"])
        return [s for p in pools for s in self.manifest["pools"][p]["subjects"]]

    def curriculum(self, subject_id: str) -> str:
        return self.sessions.loc[self.sessions["subject_id"] == subject_id, "curriculum"].iloc[0]

    def subject_sessions(self, subject_id: str, role: str | None = None) -> list[Session]:
        """A subject's sessions in date order, optionally only the ``fit`` or ``score`` ones."""
        rows = self.sessions[self.sessions["subject_id"] == subject_id].sort_values("index")
        if role is not None:
            rows = rows[rows["role"] == role]
        out = []
        for row in rows.itertuples(index=False):
            choices, rewards = self._trials[row.session_id]
            out.append(Session(subject_id=subject_id, session_id=row.session_id, index=int(row.index),
                               role=row.role, task=row.task, stage=None if pd.isna(row.stage) else row.stage,
                               choices=choices, rewards=rewards))
        return out

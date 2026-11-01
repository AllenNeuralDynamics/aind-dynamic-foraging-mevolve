"""Replay fingerprints (ADR 0001): where a program's predicted switching departs from the mice's.

On real sessions, trial t (t >= 1) is a switch when its choice differs from trial t - 1's. The
program's predicted switch probability is its probability of the other side. Trials are binned by
the history before them, three ways:

- ``losses``: consecutive unrewarded trials on the current side, 1 to 5, then 6 or more;
- ``pattern``: the last 1, 2 and 3 trials, each as (same side as the last choice?, rewarded?);
- ``run``: the last trial's reward by the current run length (1 to 5, then 6 or more).

Each statistic's gap is the trial-weighted mean of |predicted - observed| switch rate over its
bins; the replay gap is the mean of the three. Both are computed per stage group and task variant
and overall, so the feedback can say where a program loses.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

CAP = 6     # run lengths and loss counts of 6 or more share a bin


def _features(choices: np.ndarray, rewards: np.ndarray) -> dict[str, list]:
    """Bin keys for every trial t >= 1, from the trials before it."""
    out = {"losses": [], "pattern": [], "run": []}
    run, losses = 0, 0
    for t in range(1, len(choices)):
        prev = choices[t - 1]
        run = run + 1 if t > 1 and choices[t - 1] == choices[t - 2] else 1
        losses = 0 if rewards[t - 1] else (losses + 1 if t > 1 and choices[t - 1] == choices[t - 2] else 1)
        out["losses"].append(min(losses, CAP) if losses else None)
        hist = tuple((int(choices[t - lag] == prev), int(rewards[t - lag])) for lag in range(1, min(3, t) + 1))
        out["pattern"].append([hist[:n] for n in range(1, len(hist) + 1)])
        out["run"].append((int(rewards[t - 1]), min(run, CAP)))
    return out


def fingerprints(sessions, p_right: list[np.ndarray]) -> dict:
    """Gaps per statistic, overall and by (stage group, task).

    ``sessions`` carry ``choices``, ``rewards``, ``stage_group`` and ``task``; ``p_right[i]`` is the
    program's per-trial probability of a right choice on session i.
    """
    tallies = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))   # (group, stat) -> bin -> [n, pred, obs]
    for s, pr in zip(sessions, p_right):
        c, r = np.asarray(s.choices), np.asarray(s.rewards)
        if len(c) < 2:
            continue
        switch = (c[1:] != c[:-1]).astype(float)
        pred = np.where(c[:-1] == 1, 1.0 - pr[1:], pr[1:])          # p(other side)
        feats = _features(c, r)
        for group in ("all", f"{s.stage_group}|{s.task}"):
            for i in range(len(switch)):
                row = np.array([1.0, pred[i], switch[i]])
                if feats["losses"][i] is not None:
                    tallies[(group, "losses")][feats["losses"][i]] += row
                for key in feats["pattern"][i]:
                    tallies[(group, "pattern")][key] += row
                tallies[(group, "run")][feats["run"][i]] += row
    out: dict = {}
    for (group, stat), bins in tallies.items():
        n = np.array([b[0] for b in bins.values()])
        gap = np.array([abs(b[1] - b[2]) / b[0] for b in bins.values()])
        out.setdefault(group, {})[stat] = float(np.sum(n * gap) / np.sum(n))
    for group, stats in out.items():
        stats["replay_gap"] = float(np.mean([stats[k] for k in ("losses", "pattern", "run") if k in stats]))
    return out

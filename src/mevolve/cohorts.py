"""Draw the generation, validation and selection subjects from the training pool (ADR 0003).

Each cohort is stratified like the test subjects, which the wrapper chose by session-count rank
within curriculum: strata are curriculum × session count in thirds (within the curriculum), and
each cohort takes from every stratum in proportion to its size. Cohorts are drawn in order from one
seeded permutation per stratum, so they are disjoint.
Which sizes and seed to use is study 40's decision; the lists it draws are committed there.

    python -m mevolve.cohorts --freeze <dir> --studies <studies repo> --seed 0 \\
        --size generation=48 --size validation=48 --size selection=100
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def strata(subjects: dict[str, tuple[str, int]]) -> dict[tuple[str, int], list[str]]:
    """``{(curriculum, third): [subject, ...]}``; thirds by session count within the curriculum,
    0 = fewest. Ties are broken by subject id so the assignment is deterministic."""
    out: dict[tuple[str, int], list[str]] = {}
    for curriculum in sorted({c for c, _ in subjects.values()}):
        ranked = sorted((s for s, (c, _) in subjects.items() if c == curriculum),
                        key=lambda s: (subjects[s][1], s))
        for i, s in enumerate(ranked):
            out.setdefault((curriculum, 3 * i // len(ranked)), []).append(s)
    return out


def _allocate(total: int, weights: list[int]) -> list[int]:
    """Split ``total`` across strata in proportion to ``weights`` (largest remainder)."""
    exact = np.array(weights, dtype=float) * total / sum(weights)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts), kind="stable")[: total - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def _allocate_nested(total: int, keys: list[tuple[str, int]], weights: list[int]) -> list[int]:
    """Split ``total`` across curricula first, then across each curriculum's thirds, so every
    curriculum's share is as close to its share of the pool as rounding allows."""
    curricula = sorted({c for c, _ in keys})
    per_curriculum = _allocate(total, [sum(w for (c, _), w in zip(keys, weights) if c == cur)
                                       for cur in curricula])
    counts = [0] * len(keys)
    for cur, n in zip(curricula, per_curriculum):
        idx = [i for i, (c, _) in enumerate(keys) if c == cur]
        for i, k in zip(idx, _allocate(n, [weights[i] for i in idx])):
            counts[i] = k
    return counts


def draw(subjects: dict[str, tuple[str, int]], sizes: dict[str, int], seed: int,
         exclude: set[str] = frozenset()) -> dict[str, list[str]]:
    """Disjoint cohorts ``{name: [subject, ...]}``, in the order ``sizes`` lists them.

    ``subjects`` maps each training-pool subject to (curriculum, session count); ``exclude``
    subjects are never drawn.
    """
    if any(n < 0 for n in sizes.values()):
        raise ValueError(f"cohort sizes must be non-negative: {sizes}")
    groups = strata({s: v for s, v in subjects.items() if s not in exclude})
    keys = sorted(groups)
    order = {k: [groups[k][i] for i in np.random.default_rng([seed, j]).permutation(len(groups[k]))]
             for j, k in enumerate(keys)}
    if sum(sizes.values()) > sum(len(v) for v in groups.values()):
        raise ValueError("cohorts ask for more subjects than the pool holds")
    taken = {k: 0 for k in keys}
    out = {}
    for name, size in sizes.items():
        left = [len(order[k]) - taken[k] for k in keys]
        target = _allocate_nested(size, keys, [len(groups[k]) for k in keys])
        counts = [min(t, n) for t, n in zip(target, left)]
        # A stratum that runs dry passes its places to the other thirds of its own curriculum
        # first, and only what its curriculum cannot take goes to the other curricula.
        for cur in sorted({c for c, _ in keys}):
            idx = [i for i, (c, _) in enumerate(keys) if c == cur]
            short = sum(target[i] for i in idx) - sum(counts[i] for i in idx)
            for i in sorted(idx, key=lambda i: -(left[i] - counts[i])):
                add = min(short, left[i] - counts[i])
                counts[i] += add
                short -= add
        for i in np.argsort([-(n - c) for n, c in zip(left, counts)], kind="stable"):
            counts[i] += min(size - sum(counts), left[i] - counts[i])
        out[name] = [s for i, k in enumerate(keys) for s in order[k][taken[k]: taken[k] + counts[i]]]
        for i, k in enumerate(keys):
            taken[k] += counts[i]
    return out


def main(argv: list[str] | None = None) -> None:
    from .data import Freeze
    from .gate0 import gru_extra_test_subjects

    parser = argparse.ArgumentParser(description="Draw the search cohorts from the training pool")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", action="append", required=True, help="name=count, in draw order")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze = Freeze(args.freeze)
    ses = freeze.sessions[freeze.sessions["pool"] == "train"]
    counts = ses.groupby("subject_id").size()
    curricula = ses.groupby("subject_id")["curriculum"].first()
    subjects = {s: (curricula[s], int(counts[s])) for s in freeze.subjects("train")}
    exclude = set(gru_extra_test_subjects(freeze.manifest, args.studies))
    sizes = {name: int(n) for name, n in (item.split("=") for item in args.size)}
    cohorts = draw(subjects, sizes, args.seed, exclude)
    manifest_sha = hashlib.sha256((args.freeze / "manifest.json").read_bytes()).hexdigest()
    record = {"freeze_manifest_sha256": manifest_sha, "seed": args.seed, "sizes": sizes, "excluded": sorted(exclude),
              "strata": "curriculum x session-count thirds", "cohorts": cohorts}
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print({k: len(v) for k, v in cohorts.items()})


if __name__ == "__main__":
    main()

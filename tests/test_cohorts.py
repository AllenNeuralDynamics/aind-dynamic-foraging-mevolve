from collections import Counter

import pytest

from mevolve.cohorts import draw, strata

# 60 subjects in two curricula (40 + 20), session counts 10..49 and 10..29.
POOL = {**{f"a{i}": ("A", 10 + i) for i in range(40)}, **{f"b{i}": ("B", 10 + i) for i in range(20)}}
SIZES = {"generation": 12, "validation": 12, "selection": 24}


def test_cohorts_are_disjoint_and_sized():
    out = draw(POOL, SIZES, seed=0)
    assert {k: len(v) for k, v in out.items()} == SIZES
    flat = [s for v in out.values() for s in v]
    assert len(flat) == len(set(flat))


def test_each_cohort_is_stratified():
    groups = strata(POOL)
    assert len(groups) == 6 and sorted(len(v) for v in groups.values()) == [6, 7, 7, 13, 13, 14]
    where = {s: k for k, v in groups.items() for s in v}
    gen = Counter(where[s] for s in draw(POOL, SIZES, seed=0)["generation"])
    assert gen[("A", 0)] + gen[("A", 1)] + gen[("A", 2)] == 8      # 40 of 60 -> 8 of 12
    assert all(v >= 1 for v in gen.values()) and len(gen) == 6


def test_seeded_and_excluding():
    assert draw(POOL, SIZES, seed=1) == draw(POOL, SIZES, seed=1)
    assert draw(POOL, SIZES, seed=1) != draw(POOL, SIZES, seed=2)
    out = draw(POOL, SIZES, seed=0, exclude={"a0", "b5"})
    assert not {"a0", "b5"} & {s for v in out.values() for s in v}


def test_negative_size_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        draw(POOL, {"generation": 70, "validation": -20}, seed=0)


def test_too_many_is_refused():
    with pytest.raises(ValueError):
        draw(POOL, {"all": 61}, seed=0)

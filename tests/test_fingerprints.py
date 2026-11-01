from dataclasses import dataclass

import numpy as np

from mevolve.fingerprints import _features, fingerprints


@dataclass
class S:
    choices: np.ndarray
    rewards: np.ndarray
    stage_group: str = "final"
    task: str = "Coupled Baiting"


def test_features_count_losses_on_the_current_side():
    c = np.array([0, 0, 0, 1, 1])
    r = np.array([1, 0, 0, 0, 1])
    f = _features(c, r)
    assert f["losses"] == [None, 1, 2, 1]        # before trials 1..4
    assert f["run"] == [(1, 1), (0, 2), (0, 3), (0, 1)]
    assert f["pattern"][2][-1] == ((1, 0), (1, 0), (1, 1))


def test_gap_is_zero_when_predictions_equal_observed_rates():
    rng = np.random.default_rng(0)
    c = (rng.random(400) < 0.5).astype(int)
    r = (rng.random(400) < 0.4).astype(float)
    p = np.full(400, 0.5)          # with random choices the true switch rate is 0.5 in every bin
    fp = fingerprints([S(c, r)], [p])
    assert set(fp) == {"all", "final|Coupled Baiting"}
    assert fp["all"]["replay_gap"] < 0.08


def test_gap_detects_a_program_that_never_switches():
    c = np.tile([0, 1], 100)                     # the mouse always switches
    r = np.zeros(200)
    p = np.where(c == 0, 0.0, 1.0)               # predicted: always stays... p_right equals the last choice
    p = np.concatenate([[0.5], c[:-1].astype(float)])
    fp = fingerprints([S(c, r)], [p])
    assert fp["all"]["run"] > 0.99

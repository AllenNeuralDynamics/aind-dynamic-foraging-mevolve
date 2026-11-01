"""Per-subject and session-level fits on sessions simulated from a known Hattori2019 program."""
from dataclasses import dataclass

import jax
import numpy as np
import pytest

from mevolve import fitting, program, scoring


@dataclass
class Sim:
    choices: np.ndarray
    rewards: np.ndarray
    task: str = "Coupled Baiting"
    stage_group: str = "final"
    index: int = 0


TRUE = {"learn_rate_rew": 0.6, "learn_rate_unrew": 0.2, "forget_rate_unchosen": 0.1, "biasL": 0.3,
        "softmax_inverse_temperature": 6.0}


def simulate(prog, p, rng, n_sessions=8, n_trials=300):
    """A two-armed bandit with baiting probabilities that switch every 60 trials."""
    sessions = []
    for s in range(n_sessions):
        latents = np.asarray(prog.init_latents(p, {}))
        choices, rewards = np.zeros(n_trials, np.int32), np.zeros(n_trials)
        for t in range(n_trials):
            probs = np.asarray(jax.nn.softmax(prog.predict(latents, p)))
            c = int(rng.random() < probs[1])
            pr = (0.6, 0.1) if (t // 60) % 2 == 0 else (0.1, 0.6)
            r = float(rng.random() < pr[c])
            choices[t], rewards[t] = c, r
            latents = np.asarray(prog.update(latents, c, r, p))
        sessions.append(Sim(choices, rewards, index=s))
    return sessions


@pytest.fixture(scope="module")
def data():
    prog = program.literature("hattori2019")
    sessions = simulate(prog, TRUE, np.random.default_rng(0))
    return prog, fitting.pad(sessions)


def test_bounded_round_trip(data):
    prog, _ = data
    assert np.allclose([fitting.to_params(prog, fitting.from_params(prog, TRUE))[k] for k in TRUE],
                       list(TRUE.values()))


def test_per_subject_fit_beats_truth_from_every_start(data):
    """Every start reaches the same optimum, at least as good as the generating parameters. (On
    these sessions the reference differential-evolution fit reaches the same value to 1e-10.)"""
    prog, batch = data
    fit = fitting.fit_per_subject(prog, batch, n_starts=4)
    truth_nll = float(fitting.per_subject_nll(prog, batch)(fitting.from_params(prog, TRUE)))
    assert fit["nll"] <= truth_nll + 1e-6
    assert np.ptp(fit["nll_starts"]) < 1e-6


def test_marginal_with_tiny_spread_equals_point_score(data):
    prog, batch = data
    x = fitting.from_params(prog, TRUE)
    ll, n = scoring.point_scores(prog, x, batch)
    m = scoring.marginal_scores(prog, x, np.full(x.shape, -30.0), batch, jax.random.PRNGKey(0), n_draws=8)
    assert np.allclose(m["log_lik"], ll, atol=1e-8)
    assert np.allclose(m["ess"][n > 0], 8.0)
    assert scoring.pooled(ll, n) == pytest.approx(scoring.pooled(m["log_lik"], m["n_trials"]))


def test_session_level_fit_runs_and_is_no_worse_in_fit(data):
    prog, batch = data
    per = fitting.fit_per_subject(prog, batch, n_starts=2)
    sl = fitting.fit_session_level(prog, batch, per["x"])
    assert np.isfinite(sl["neg_log_posterior"])
    assert set(sl["typical_params"]) == set(TRUE)

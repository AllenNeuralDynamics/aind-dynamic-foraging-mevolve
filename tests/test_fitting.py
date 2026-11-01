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


def test_session_level_recovers_a_known_spread():
    """Sessions whose learning rate varies with a known spread: Laplace-EM recovers that spread
    to within a factor of 2 and keeps the fixed parameters' spreads small."""
    prog = program.literature("hattori2019")
    rng = np.random.default_rng(3)
    x_true = fitting.from_params(prog, TRUE)
    spread = np.array([0.6, 0.0, 0.0, 0.0, 0.0])
    sessions = []
    for s in range(24):
        p = {k: float(v) for k, v in fitting.to_params(prog, x_true + spread * rng.normal(size=5)).items()}
        sessions += simulate(prog, p, rng, n_sessions=1, n_trials=400)
    batch = fitting.pad(sessions)
    per = fitting.fit_per_subject(prog, batch, n_starts=2)
    sl = fitting.fit_session_level(prog, batch, per["x"], max_em=100)
    sigma = np.exp(np.asarray(sl["log_sigma"]))
    assert sl["converged"]
    assert 0.3 < sigma[0] < 1.2, sigma
    assert np.all(sigma[1:] < 0.6), sigma


def test_marginal_p_right_with_tiny_spread_equals_point(data):
    prog, batch = data
    x = fitting.from_params(prog, TRUE)
    point = scoring.point_p_right(prog, x, batch)
    marg = scoring.marginal_p_right(prog, x, np.full(x.shape, -30.0), batch, jax.random.PRNGKey(1), n_draws=4)
    m = np.asarray(batch.mask)
    assert np.allclose(point[m], marg[m], atol=1e-8)


def test_m_step_finds_the_spread_when_sessions_agree():
    """Review case: 24 sessions at theta = 0 with variance 0.001 each, from log sigma = -1; the
    optimum of the log-sigma objective is at about -3.401 (Newton steps overshot to -9)."""
    import jax.numpy as jnp
    theta = jnp.zeros((24, 1)); var = jnp.full((24, 1), 0.001); live = jnp.ones(24, bool)
    _, u = fitting._m_step(theta, var, live, jnp.array([-1.0]))
    assert float(u[0]) == pytest.approx(-3.401, abs=2e-3)


def test_a_zero_probability_costs_what_the_reference_charges():
    import jax.numpy as jnp
    prog = program.literature("loss_counting")
    x = fitting.from_params(prog, {"loss_count_threshold_mean": 1.0, "loss_count_threshold_std": 0.0, "biasL": 0.0})
    sim = [Sim(np.array([0, 0, 0], np.int32), np.array([0.0, 0.0, 0.0]))]     # stays after two losses
    batch = fitting.pad(sim)
    nll = float(fitting.per_subject_nll(prog, batch)(x))
    assert nll < 100        # 0.5 on trial 0, then zero-probability stays at -log(1e-16) each, not 691


def test_a_program_with_no_parameters_is_rejected(tmp_path):
    from mevolve.program import load
    path = tmp_path / "fixed.py"
    path.write_text("PARAMS = {}\nN_LATENT = 0\nLATENT_NAMES = ()\n"
                    "def init_latents(p, session): ...\ndef predict(latents, p): ...\n"
                    "def update(latents, choice, reward, p): ...\n")
    with pytest.raises(ValueError, match="PARAMS is empty"):
        load(path)

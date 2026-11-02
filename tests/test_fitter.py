"""The fitter's public interface: parameters under each regime, and latent variables on every trial."""
import jax
import numpy as np
import pytest

from mevolve import evaluate, fitter, fitting, program, scoring

from test_fitting import TRUE, simulate
from test_score import _subject


@pytest.fixture(scope="module")
def sessions():
    prog = program.literature("hattori2019")
    return prog, simulate(prog, TRUE, np.random.default_rng(1), n_sessions=4, n_trials=150)


def test_per_subject_fit_and_its_latents(sessions):
    prog, sess = sessions
    f = fitter.fit(prog, sess, "subject", n_starts=2)
    assert f.regime == "subject" and f.mu is None and set(f.params) == set(prog.params)
    lat = f.latents()
    assert len(lat) == len(sess)
    logp = np.asarray(fitting.batch_logp(prog, f.x, f.batch))
    for i, (s, out) in enumerate(zip(sess, lat)):
        assert out["latents"].shape == (len(s.choices), len(prog.latent_names))
        p_chosen = out["probs"][np.arange(len(s.choices)), s.choices]
        np.testing.assert_allclose(np.log(p_chosen), logp[i, : len(s.choices)], atol=1e-10)


def test_two_level_fit_gives_each_session_its_own_parameters(sessions):
    prog, sess = sessions
    f = fitter.fit(prog, sess, "2level", n_starts=2, max_em=3)
    assert 1 <= f.em_iterations <= 3 and isinstance(f.converged, bool)
    assert set(f.typical_params) == set(prog.params)
    per_session = f.session_params()
    assert len(per_session) == len(sess) and len(f.latents()) == len(sess)
    assert len({round(p["learn_rate_rew"], 12) for p in per_session}) > 1    # sessions differ


def test_unknown_regime_is_refused(sessions):
    prog, sess = sessions
    with pytest.raises(ValueError, match="regime"):
        fitter.fit(prog, sess, "session")


def test_evaluator_scores_held_out_sessions_at_the_fitters_parameters():
    """The evaluator adds scoring on top of the fitter and changes no number."""
    d = _subject(0)
    prog = program.literature("hattori2019")
    rec, = evaluate.evaluate(program.LITERATURE_DIR / "hattori2019.py", [d], regime="subject", n_starts=2, workers=1)
    f = fitter.fit(prog, d.fit, "subject", n_starts=2)
    ll, n = scoring.point_scores(prog, f.x, fitting.pad(d.score))
    assert rec["log_lik"] == pytest.approx(float(ll[: len(d.score)].sum()), abs=1e-12)
    assert rec["params"] == f.params

"""Gate 1a on synthetic sessions: every literature program against its reference forager."""
import numpy as np
import pytest

from mevolve import literature, parity, program

AVAILABLE = list(literature.LITERATURE)     # every registered model: a missing program fails


@pytest.mark.parametrize("name", AVAILABLE)
def test_program_params_equal_the_reference_fit(name):
    free, _ = literature.fit_spec(name)
    assert program.literature(literature.LITERATURE[name][0]).params == free


@pytest.mark.parametrize("name", AVAILABLE)
def test_program_matches_reference(name):
    rng = np.random.default_rng(0)
    choices, rewards = parity.synthetic_sessions(rng)
    for _ in range(5):
        result = parity.compare(name, parity.draw_params(name, rng), choices, rewards)
        assert result["ok"], result


def test_reference_logits_match_the_programs_on_softmax_models():
    """The exclusion rule reads the reference's logits; on ordinary draws they equal the program's."""
    from mevolve.parity import _reference_max_logit
    rng = np.random.default_rng(1)
    choices, rewards = parity.synthetic_sessions(rng, n_sessions=1)
    for name in parity.REFERENCE_SOFTMAX:
        params = parity.draw_params(name, rng)
        _, clamped = literature.fit_spec(name)
        ref = literature.oracle(name, params={**params, **clamped})
        ref.perform_closed_loop(choices[0], rewards[0])
        prog = program.literature(literature.LITERATURE[name][0])
        _, latents = program.replay(prog, params, choices[0], rewards[0])
        ours = np.asarray([np.max(prog.predict(lat, {k: np.float64(v) for k, v in params.items()})) for lat in latents])
        assert np.allclose(_reference_max_logit(name, ref), ours, atol=1e-6)


def test_epsilon_greedy_exact_tie_survives_jit():
    """Three rewarded left choices make Q(left) - 0.488 exactly 0 in numpy; a fused multiply-add
    under jit would break the tie and move p(left) from 0.5 to 0.9 (found in review)."""
    choices = [np.array([0, 0, 0, 0, 1], dtype=np.int64)]
    rewards = [np.array([1.0, 1.0, 1.0, 0.0, 0.0])]
    result = parity.compare("Rescorla-Wagner", {"learn_rate": 0.2, "biasL": -0.488, "epsilon": 0.2}, choices, rewards)
    assert result["ok"], result


def test_mismatched_sessions_are_refused():
    c = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)]
    with pytest.raises(ValueError, match="same sessions"):
        parity.compare("Hattori2019", parity.draw_params("Hattori2019", np.random.default_rng(0)), c, [np.zeros(2)])
    with pytest.raises(ValueError, match="same sessions"):
        parity.compare("Hattori2019", parity.draw_params("Hattori2019", np.random.default_rng(0)), c[:1], [np.zeros(3)])


def test_gate1a_study12_needs_a_source(tmp_path):
    from mevolve import gate1a
    with pytest.raises(SystemExit):
        gate1a.main(["--freeze", str(tmp_path), "--out", str(tmp_path / "x.json"), "--params", "study12"])

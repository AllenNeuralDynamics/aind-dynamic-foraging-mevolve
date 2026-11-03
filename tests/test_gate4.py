"""Gate 4's simulator and check, without a freeze."""
import numpy as np

from mevolve import fitting, gate4, program
from mevolve.data import Session


def _real(n_trials):
    return [Session("s1", f"a{i}", i, "fit", "Uncoupled Without Baiting", "STAGE_FINAL",
                    np.zeros(n, np.int32), np.zeros(n)) for i, n in enumerate(n_trials)]


def test_simulated_sessions_keep_the_real_lengths_and_are_labelled_coupled_baiting():
    prog = program.literature("hattori2019")
    mu = np.asarray(fitting.from_params(prog, {"learn_rate_rew": 0.5, "learn_rate_unrew": 0.2,
                                               "forget_rate_unchosen": 0.3, "biasL": 0.0,
                                               "softmax_inverse_temperature": 5.0}))
    sessions, params = gate4.simulate(prog, mu, np.full(5, -1.0), _real([50, 80]), np.random.default_rng(0),
                                      gate4._stepper(prog))
    assert [len(s.choices) for s in sessions] == [50, 80]
    assert all(s.task == "Coupled Baiting" and s.session_id.startswith("sim-") for s in sessions)
    assert params[0] != params[1]          # each session draws its own parameters
    assert all(set(np.unique(s.rewards)) <= {0.0, 1.0} for s in sessions)


def test_check_passes_when_the_fit_returns_the_truth_and_fails_when_spreads_are_off():
    prog = program.literature("hattori2019")
    true = {f"{i}": (np.array([0.1, -1.2, 0.5, 0.0, -1.8]) + 0.1 * i, np.log([0.3, 0.4, 0.4, 0.1, 0.1]))
            for i in range(4)}
    rec = lambda s, scale: {"subject_id": f"sim-{s}", "converged": True, "n_trials": 100,       # noqa: E731
                            "log_lik": -60.0, "per_subject_log_lik": -61.0,
                            "typical_params": {k: float(v) for k, v in fitting.to_params(prog, true[s][0]).items()},
                            "log_sigma": (true[s][1] + np.log(scale)).tolist()}
    assert gate4.check(prog, true, [rec(s, 1.0) for s in true])["ok"]
    bad = gate4.check(prog, true, [rec(s, 3.0) for s in true])
    assert not bad["ok"] and bad["params"]["learn_rate_rew"]["sigma_ok"] is False

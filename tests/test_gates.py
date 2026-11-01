import numpy as np
import pytest

from mevolve.gate1 import compare
from mevolve.gate2 import paired_se


def _rec(s, ll, n):
    return {"subject_id": s, "log_lik": ll, "n_trials": n}


def test_compare_pools_by_trials_and_diffs_per_subject():
    recs = [_rec("a", np.log(0.7) * 100, 100), _rec("b", np.log(0.6) * 300, 300)]
    c = compare(recs, {"a": 0.69, "b": 0.6}, 0.62)
    assert c["pooled"] == pytest.approx(np.exp((np.log(0.7) * 100 + np.log(0.6) * 300) / 400))
    assert c["per_subject_diff"]["max"] == pytest.approx(0.01)
    assert c["subjects_worse_than_reference_by_0.005"] == []


def test_paired_se_is_zero_for_a_constant_shift_and_signs_a_minus_b():
    a = [_rec(str(i), -100.0 - i, 200) for i in range(10)]
    b = [_rec(str(i), -110.0 - i, 200) for i in range(10)]
    out = paired_se(a, b)
    assert out["score_diff_x100"] > 0 and out["se_x100"] == pytest.approx(0.0, abs=1e-12)


def test_paired_se_ignores_subject_length():
    """Review case: 100, 200 and 400 trials at -0.50 per trial for A and -0.55 for B."""
    a = [_rec(str(n), -0.50 * n, n) for n in (100, 200, 400)]
    b = [_rec(str(n), -0.55 * n, n) for n in (100, 200, 400)]
    assert paired_se(a, b)["se_x100"] == pytest.approx(0.0, abs=1e-9)


def test_compare_study12_passes_equal_fits_and_fails_a_worse_one():
    from mevolve.gate1 import compare_study12
    recs = [{"subject_id": str(i), "log_lik": -60.0, "n_trials": 100, "fit_log_lik": -70.0} for i in range(20)]
    ref = {str(i): {"eval_ll": -60.0, "eval_n": 100, "fit_ll": -70.0} for i in range(20)}
    assert compare_study12(recs, ref)["ok"]
    worse = [dict(r, fit_log_lik=-75.0) if i < 2 else r for i, r in enumerate(recs)]   # 10% worse by 5 nats
    assert not compare_study12(worse, ref)["ok"]


def test_compare_study12_fails_on_missing_reference_subjects():
    from mevolve.gate1 import compare_study12
    recs = [{"subject_id": str(i), "log_lik": -60.0, "n_trials": 100, "fit_log_lik": -70.0} for i in range(3)]
    ref = {"0": {"eval_ll": -60.0, "eval_n": 100, "fit_ll": -70.0}}
    out = compare_study12(recs, ref)
    assert not out["ok"] and out["subjects_missing_from_reference"] == 2

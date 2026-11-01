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

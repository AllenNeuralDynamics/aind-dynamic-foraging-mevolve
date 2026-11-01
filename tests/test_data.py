import pytest

from mevolve.data import Freeze


def test_sessions_in_order_with_ignored_trials_removed(freeze_dir):
    freeze = Freeze(freeze_dir)
    sessions = freeze.subject_sessions("t2")
    assert [s.index for s in sessions] == [0, 1, 2, 3]
    assert [s.role for s in sessions] == ["fit", "score", "fit", "score"]
    assert sessions[0].choices.tolist() == [0, 1, 1, 0]          # response 2 dropped
    assert sessions[0].rewards.tolist() == [0.0, 1.0, 1.0, 0.0]  # its reward with it
    assert [s.stage_group for s in sessions] == ["unknown", "final", "final", "final"]


def test_role_filter_and_subject_lists(freeze_dir):
    freeze = Freeze(freeze_dir)
    assert [s.index for s in freeze.subject_sessions("h1", role="score")] == [1, 3]
    assert freeze.subjects("heldout") == ["h1", "h2"]
    assert freeze.subjects() == ["h1", "h2", "t1", "t2", "t3"]


def test_digest_mismatch_is_refused(freeze_dir):
    (freeze_dir / "trials.parquet").write_bytes((freeze_dir / "trials.parquet").read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        Freeze(freeze_dir)

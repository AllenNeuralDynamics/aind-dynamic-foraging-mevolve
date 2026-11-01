"""Gate 1a on synthetic sessions: every literature program against its reference forager."""
import numpy as np
import pytest

from mevolve import literature, parity, program

AVAILABLE = [n for n, (f, _, _) in literature.LITERATURE.items() if (program.LITERATURE_DIR / f"{f}.py").exists()]


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

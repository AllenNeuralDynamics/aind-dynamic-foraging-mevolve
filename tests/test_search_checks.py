"""Static checks: every literature program passes; unsafe or oversized candidates are refused."""
import pytest

from mevolve import program
from mevolve.search.checks import ProgramRejected, check_source

GOOD = '''import jax.numpy as jnp
PARAMS = {"bias": (-5.0, 5.0)}
N_LATENT = 1
LATENT_NAMES = ("last_choice",)
def init_latents(p, session): return jnp.zeros(1)
def predict(latents, p): return jnp.array([0.0, p["bias"]])
def update(latents, choice, reward, p): return latents
'''


@pytest.mark.parametrize("path", sorted(program.LITERATURE_DIR.glob("*.py")), ids=lambda p: p.stem)
def test_every_literature_program_passes(path):
    out = check_source(path.read_text())
    assert out["n_params"] <= 8 and out["n_latent"] <= 6


def test_a_minimal_program_passes():
    assert check_source(GOOD) == {"n_params": 1, "n_latent": 1}


@pytest.mark.parametrize("change, reason", [
    (("import jax.numpy as jnp", "import os"), "only jax"),
    (("import jax.numpy as jnp", "from pathlib import Path"), "only jax"),
    (("return latents", "return open('x').read()"), "'open'"),
    (("return latents", "return latents.__class__"), "__class__"),
    (("return latents", "return __import__('os')"), "__import__"),
    (("return latents", "return getattr(jnp, 'zeros')(1)"), "getattr"),
    (("return latents", "return __builtins__['open']('x')"), "__builtins__"),
    (("import jax.numpy as jnp", "import jax.numpy as __jnp"), "__jnp"),
    (("import jax.numpy as jnp", "from jax import __version__"), "__version__"),
    (("return latents", "jax.config.update('jax_enable_x64', False)"), "jax.config"),
    (('"bias": (-5.0, 5.0)', '"bias": (5.0, -5.0)'), "low < high"),
    (("N_LATENT = 1", "N_LATENT = 7"), "1 to 6"),
    (("N_LATENT = 1", "N_LATENT = 2"), "LATENT_NAMES has 1"),
    (("def update(", "def step("), "update is not defined"),
    (("return latents", "return latents +"), "syntax error"),
])
def test_unsafe_or_malformed_programs_are_refused(change, reason):
    old, new = change
    with pytest.raises(ProgramRejected, match=reason):
        check_source(GOOD.replace(old, new))


def test_more_than_eight_parameters_is_refused():
    params = ", ".join(f'"p{i}": (0.0, 1.0)' for i in range(9))
    with pytest.raises(ProgramRejected, match="at most 8"):
        check_source(GOOD.replace('{"bias": (-5.0, 5.0)}', "{" + params + "}"))


def test_computed_params_are_refused():
    with pytest.raises(ProgramRejected, match="plain literals"):
        check_source(GOOD.replace('{"bias": (-5.0, 5.0)}', "dict(bias=(-5.0, 5.0))"))

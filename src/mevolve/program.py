"""Programs: the unit the search evolves, and their replay through a real session.

A program is a Python module (one file, so the LLM can read and rewrite it) that defines:

    PARAMS = {"name": (low, high), ...}     # fitted by the evaluator, never chosen by the LLM
    N_LATENT = 2                            # latent variables, reset at the start of each session
    LATENT_NAMES = ("q_left", "q_right")
    def init_latents(p, session): ...       # -> array (N_LATENT,)
    def predict(latents, p): ...            # -> logits over [left, right] (log-probabilities work too)
    def update(latents, choice, reward, p): ...   # -> array (N_LATENT,)

``p`` maps each parameter name to a scalar; ``session`` maps each session fact (``stage``,
``coupled``, ``baited``, ``index``) to a scalar. Everything runs in float64, so a program can match
the numpy reference implementations to 1e-6 per trial.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

SESSION_FACTS = ("stage", "coupled", "baited", "index")


@dataclass(frozen=True)
class Program:
    name: str
    source: str
    params: dict[str, tuple[float, float]]
    latent_names: tuple[str, ...]
    init_latents: Callable
    predict: Callable
    update: Callable

    @property
    def n_latent(self) -> int:
        return len(self.latent_names)


def load(path: Path | str) -> Program:
    """Load a program from its source file and check that it defines the interface."""
    path = Path(path)
    source = path.read_text()
    module = types.ModuleType(path.stem)
    exec(compile(source, str(path), "exec"), module.__dict__)
    missing = [n for n in ("PARAMS", "N_LATENT", "LATENT_NAMES", "init_latents", "predict", "update")
               if not hasattr(module, n)]
    if missing:
        raise ValueError(f"{path.name} does not define {', '.join(missing)}")
    if len(module.LATENT_NAMES) != module.N_LATENT:
        raise ValueError(f"{path.name}: N_LATENT is {module.N_LATENT} but LATENT_NAMES has {len(module.LATENT_NAMES)}")
    if not module.PARAMS:       # every regime fits at least one parameter
        raise ValueError(f"{path.name}: PARAMS is empty; a program needs at least one fitted parameter")
    params = {k: (float(lo), float(hi)) for k, (lo, hi) in module.PARAMS.items()}
    return Program(name=path.stem, source=source, params=params, latent_names=tuple(module.LATENT_NAMES),
                   init_latents=module.init_latents, predict=module.predict, update=module.update)


def replay(program: Program, p: dict, choices, rewards, session: dict | None = None):
    """Run a program through one session's real choices and rewards.

    Returns the choice probabilities, shape (n_trials, 2), and the latent variables before each
    trial, shape (n_trials, N_LATENT). Trial t is predicted from the latent variables after
    trials 0 .. t-1, so trial 0 is predicted from ``init_latents``.
    """
    session = session or {k: 0 for k in SESSION_FACTS}
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()}
    session = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in session.items()}

    def step(latents, trial):
        choice, reward = trial
        probs = jax.nn.softmax(jnp.asarray(program.predict(latents, p), dtype=jnp.float64))
        return program.update(latents, choice, reward, p), (probs, latents)

    latents0 = jnp.asarray(program.init_latents(p, session), dtype=jnp.float64)
    _, (probs, latents) = jax.lax.scan(step, latents0, (jnp.asarray(choices, dtype=jnp.int32),
                                                        jnp.asarray(rewards, dtype=jnp.float64)))
    return probs, latents


LITERATURE_DIR = Path(__file__).parent / "programs" / "literature"


def literature(name: str) -> Program:
    """One literature model as a program, by its module name in ``programs/literature``."""
    return load(LITERATURE_DIR / f"{name}.py")

"""The task message: what Claude reads at the start of every proposal (Shinka's ``task_sys_msg``).

It is the same for every proposal, so the shim caches it (``claude.Settings.cache_prefix``). It
holds the task, the program interface, how programs are fitted and scored, what makes a good
program, and the whole literature library as reference code (study 40 ADR 0006). Shinka appends
the edit format after it; the parent, the inspirations and the feedback go in the user message.
"""
from __future__ import annotations

from ..literature import LITERATURE
from ..program import LITERATURE_DIR

TASK = """\
You are improving a cognitive model of how mice make choices in a dynamic foraging task. Each \
candidate is a short JAX program. We fit its parameters to each mouse and score how well it \
predicts choices in sessions it was not fitted on. Change the program's mechanism so that it \
predicts better, while it stays a model a neuroscientist can read and interpret.

# The task

On each trial a mouse licks the left or the right spout. Each side gives water with a probability \
that changes, without warning, in blocks of tens of trials. In coupled tasks the two probabilities \
move together; in uncoupled tasks they change independently. In baited tasks a reward assigned to \
the side the mouse did not choose stays there until collected, so a side left alone grows more \
likely to pay. Mice learn the task over weeks of training stages.

# What a program sees

Each session's choices (0 = left, 1 = right) and rewards (0 or 1), trial by trial; trials the \
mouse ignored are removed. At the start of each session it gets four session facts: `stage` (-1 \
unknown, 0 earlier, 1 final), `coupled` (1 or 0), `baited` (1 or 0), and `index`, the session's \
position in the mouse's training history (0, 1, 2, ...). Latent variables reset at the start of \
every session.

# The interface

```python
import jax.numpy as jnp
PARAMS = {"learn_rate": (0.0, 1.0), "beta": (0.0, 20.0)}   # name: (low, high); at most 8
N_LATENT = 2                                               # at most 6
LATENT_NAMES = ("q_left", "q_right")
def init_latents(p, session): ...            # -> array of N_LATENT values
def predict(latents, p): ...                 # -> logits over [left, right]
def update(latents, choice, reward, p): ...  # -> array of N_LATENT values
```

`p` maps each parameter name to a number and `session` maps each fact to a number. Write pure \
`jax.numpy` code: no Python `if` on values (use `jnp.where`), no loops over trials (we run \
`update` over the session), no randomness, no files or network, no imports beyond `jax` and \
`math`. PARAMS, N_LATENT and LATENT_NAMES must be plain literals.

# How a program is fitted and scored

You choose the structure and each parameter's bounds, never the values. We fit a 2-level model: \
each session's parameters vary around the mouse's typical values. A held-out session is scored by \
averaging predictions over parameter sets drawn for that mouse, each weighted by how well it \
explains the session's earlier trials. The score is the per-trial likelihood of held-out choices, \
pooled over the validation mice, x100. Higher is better, and 50 is chance. Gains smaller than about \
0.05 may be noise. Choose bounds that contain plausible values without being needlessly wide. If \
the 2-level fit fails for a program, it is scored with one parameter set per mouse, and the \
feedback says so.

# A good program

At most 8 parameters and 6 latent variables; larger programs are rejected before fitting. Give \
every parameter and latent variable a name and a role a neuroscientist would recognise: a learning \
rate, a choice trace, a belief about the current block. Prefer one clear mechanism to several small \
corrections. Parameters that do not change the predictions, or that always fit at a bound, are \
flagged; remove them or fix their role. Use `index` only for a mechanism that changes with \
experience, never to tell sessions apart.

# Feedback

For each program you see its score, the score by stage and task, its fitted parameters, and how \
its predicted switching departs from the mice's after the same recent history. These details come \
from a separate group of mice, the generation mice. Use them to find what the program gets \
systematically wrong.

# Your answer

In the DESCRIPTION, state first, in one or two sentences, the cognitive hypothesis behind your \
change. Then give the new program in the format below.
"""

LIBRARY_HEADER = """
# Literature models, for reference

These are models from the literature, written in the same interface. Borrow, combine or simplify \
their mechanisms; do not copy one unchanged.
"""


def library(exclude: tuple[str, ...] = ()) -> str:
    """Every literature program's source, in study 12's naming, alphabetically, except ``exclude``."""
    unknown = set(exclude) - set(LITERATURE)
    if unknown:
        raise ValueError(f"not literature models: {sorted(unknown)}")
    parts = [LIBRARY_HEADER]
    for name in sorted(set(LITERATURE) - set(exclude)):
        source = (LITERATURE_DIR / f"{LITERATURE[name][0]}.py").read_text().strip()
        parts.append(f"\n## {name}\n\n```python\n{source}\n```\n")
    return "".join(parts)


LIMIT_TEXT = ("# name: (low, high); at most {p}", "# at most {l}", "At most {p} parameters and {l} latent variables")


def task_message(with_library: bool = True, exclude: tuple[str, ...] = (), max_params: int = 8,
                 max_latent: int = 6) -> str:
    """The task, with the run's program limits (``checks.check_source``), then the library."""
    task = TASK
    for text in LIMIT_TEXT:
        old = text.format(p=8, l=6)
        assert task.count(old) == 1, old
        task = task.replace(old, text.format(p=max_params, l=max_latent))
    return task + (library(tuple(exclude)) if with_library else "")

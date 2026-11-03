"""Static checks on a candidate program, before any of it runs (study 40 ADR 0005).

A candidate is code Claude wrote, and loading it executes it. So its source is parsed, never run,
until it passes: imports only from jax and math, no file, process or introspection access, no
dunder names or attributes (``__builtins__["open"]`` would otherwise reach the real builtins). The interface limits (at most 8 parameters, at most 6 latent variables) are read
from the parsed literals too, so an oversized program is rejected before fitting.
"""
from __future__ import annotations

import ast

MAX_PARAMS, MAX_LATENT = 8, 6
ALLOWED_IMPORTS = ("jax", "math")       # and their submodules
FORBIDDEN_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                             "getattr", "setattr", "delattr", "input", "breakpoint", "help", "memoryview",
                             "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
                             "builtins", "pickle", "ctypes"})


class ProgramRejected(ValueError):
    """A candidate that fails a static check; the message says which and why."""


def _module_allowed(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in ALLOWED_IMPORTS)


def check_source(source: str) -> dict:
    """Parse and check ``source``; returns ``{"n_params", "n_latent"}`` or raises
    ``ProgramRejected``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ProgramRejected(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if not _module_allowed(a.name)]
            if bad:
                raise ProgramRejected(f"import of {', '.join(bad)}: only jax and math may be imported")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _module_allowed(node.module or ""):
                raise ProgramRejected(f"import from {node.module!r}: only jax and math may be imported")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ProgramRejected(f"use of {node.id!r} is not allowed in a program")
        elif isinstance(node, ast.alias) and (node.name.startswith("__") or (node.asname or "").startswith("__")):
            raise ProgramRejected(f"import of {node.name!r} as {node.asname!r}: dunder names are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ProgramRejected(f"access to {node.attr!r} is not allowed in a program")
        elif isinstance(node, ast.Attribute) and node.attr == "config" and _root_name(node.value) == "jax":
            raise ProgramRejected("a program may not change jax.config")
    assigned = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    for name in ("PARAMS", "N_LATENT", "LATENT_NAMES"):
        if name not in assigned:
            raise ProgramRejected(f"{name} is not assigned at the top level")
    try:
        params = ast.literal_eval(assigned["PARAMS"])
        n_latent = ast.literal_eval(assigned["N_LATENT"])
        names = ast.literal_eval(assigned["LATENT_NAMES"])
    except ValueError as e:
        raise ProgramRejected("PARAMS, N_LATENT and LATENT_NAMES must be plain literals") from e
    if not isinstance(params, dict) or not params:
        raise ProgramRejected("PARAMS must be a non-empty dict of name: (low, high)")
    for k, v in params.items():
        if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)
                and v[0] < v[1]):
            raise ProgramRejected(f"PARAMS[{k!r}] must be (low, high) numbers with low < high")
    if len(params) > MAX_PARAMS:
        raise ProgramRejected(f"{len(params)} parameters; at most {MAX_PARAMS} are allowed")
    if not isinstance(n_latent, int) or not 0 < n_latent <= MAX_LATENT:
        raise ProgramRejected(f"N_LATENT is {n_latent!r}; it must be 1 to {MAX_LATENT}")
    if len(names) != n_latent:
        raise ProgramRejected(f"N_LATENT is {n_latent} but LATENT_NAMES has {len(names)} names")
    for fn in ("init_latents", "predict", "update"):
        if not any(isinstance(n, ast.FunctionDef) and n.name == fn for n in tree.body):
            raise ProgramRejected(f"{fn} is not defined at the top level")
    return {"n_params": len(params), "n_latent": n_latent}


def _root_name(node) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None

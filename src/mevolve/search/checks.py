"""Static checks on a candidate program, before any of it runs (study 40 ADR 0005).

A candidate is code Claude wrote, and loading it executes it. So its source is parsed, never run,
until it passes: imports only from jax and math, no file, process or introspection access, no
dunder names, no private attributes (``__builtins__["open"]`` or ``license._Printer__filenames`` would
otherwise reach the real builtins or the disk), no frame attributes or generators (``gi_frame.f_builtins``),
and none of jax's file, callback or debugging functions (``jnp.save``, ``jax.pure_callback``). The interface
limits (by default at most 8 parameters and 6 latent variables; a run can set others, #60) are read
from the parsed literals too, so an oversized program is rejected before fitting.

These checks keep a cooperative proposer inside the program interface; they are not a sandbox against
code written to escape. The evaluation script also removes API keys from the environment before any
candidate runs.
"""
from __future__ import annotations

import ast

MAX_PARAMS, MAX_LATENT = 8, 6
ALLOWED_IMPORTS = ("jax", "math")       # and their submodules
FORBIDDEN_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                             "getattr", "setattr", "delattr", "input", "breakpoint", "help", "memoryview",
                             "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
                             "builtins", "pickle", "ctypes", "license", "credits", "copyright", "exit", "quit"})
# Attribute names a program never needs: frames and code objects, and jax/numpy file IO, host
# callbacks, debugging and configuration. Checked by name, whatever module or alias they hang off.
FORBIDDEN_ATTRS = frozenset({"gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "f_builtins",
                             "f_globals", "f_locals", "f_back", "f_code", "tb_frame", "tb_next", "co_code",
                             "save", "savez", "savez_compressed", "load", "fromfile", "tofile", "loadtxt",
                             "savetxt", "genfromtxt", "memmap", "fromregex", "callback", "pure_callback",
                             "io_callback", "debug", "profiler", "distributed", "experimental", "config",
                             "format", "format_map"})


class ProgramRejected(ValueError):
    """A candidate that fails a static check; the message says which and why."""


def _module_allowed(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in ALLOWED_IMPORTS)


def check_source(source: str, max_params: int = MAX_PARAMS, max_latent: int = MAX_LATENT) -> dict:
    """Parse and check ``source``; returns ``{"n_params", "n_latent"}`` or raises
    ``ProgramRejected``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ProgramRejected(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if not _module_allowed(a.name)
                   or FORBIDDEN_ATTRS.intersection(a.name.split("."))]
            if bad:
                raise ProgramRejected(f"import of {', '.join(bad)}: only jax and math may be imported, "
                                      "without their file, callback or debugging parts")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _module_allowed(node.module or ""):
                raise ProgramRejected(f"import from {node.module!r}: only jax and math may be imported")
            bad = [a.name for a in node.names
                   if a.name in FORBIDDEN_ATTRS or FORBIDDEN_ATTRS.intersection((node.module or "").split("."))]
            if bad:
                raise ProgramRejected(f"import of {', '.join(bad)} from {node.module!r} is not allowed in a program")
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFunctionDef)):
            raise ProgramRejected("generators and async code are not allowed in a program")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ProgramRejected(f"use of {node.id!r} is not allowed in a program")
        elif isinstance(node, ast.alias) and (node.name.startswith("__") or (node.asname or "").startswith("__")):
            raise ProgramRejected(f"import of {node.name!r} as {node.asname!r}: dunder names are not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS):
            raise ProgramRejected(f"access to {node.attr!r} is not allowed in a program")
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
    if len(params) > max_params:
        raise ProgramRejected(f"{len(params)} parameters; at most {max_params} are allowed")
    if not isinstance(n_latent, int) or not 0 < n_latent <= max_latent:
        raise ProgramRejected(f"N_LATENT is {n_latent!r}; it must be 1 to {max_latent}")
    if len(names) != n_latent:
        raise ProgramRejected(f"N_LATENT is {n_latent} but LATENT_NAMES has {len(names)} names")
    for fn in ("init_latents", "predict", "update"):
        if not any(isinstance(n, ast.FunctionDef) and n.name == fn for n in tree.body):
            raise ProgramRejected(f"{fn} is not defined at the top level")
    return {"n_params": len(params), "n_latent": n_latent}

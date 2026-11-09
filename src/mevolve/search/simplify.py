"""Simplification mode (ADR 0006): Halstead effort and the hard floor.

A simplification run rewrites programs that already score well so that they are simpler. A candidate
is correct only if its validation score (x100, the search's evaluator) is at or above the run's
floor; its ``combined_score`` is then minus its Halstead effort, so Shinka, which maximizes, prefers
the simpler program. A program below the floor is incorrect but keeps its metrics.

The evaluation script applies the floor to the search's metrics after fitting, or after a repeat-cache
hit, so both caches hold the search's metrics and are shared by search and simplify runs.
"""
from __future__ import annotations

import ast
import math


def _is_bare_string(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def halstead_effort(source: str) -> float:
    """The program's Halstead effort, by this rule:

    1. Parse the module's source (``ast.parse``). Comments, formatting and the evolve-block markers
       are not in the syntax tree, so they never count.
    2. Drop every bare string statement (docstrings included) from every statement list.
    3. Return radon's Halstead total for the module, ``radon.metrics.h_visit_ast(tree).total.effort``
       (radon 6.0.1): effort = difficulty x volume, with difficulty = (h1 / 2) x (N2 / h2) and volume
       = (N1 + N2) x log2(h1 + h2), over the module's operators (h1 distinct, N1 in all) and operands
       (h2, N2). Radon counts as operators the arithmetic, boolean, comparison and augmented-assignment
       operators, unary ones included (``-5.0`` in a bound is one), and as operands what they act on;
       calls, subscripts and assignments alone add nothing.

    Deterministic: the same source always gives the same number. Raises ``SyntaxError`` on code that
    does not parse."""
    from radon.metrics import h_visit_ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                setattr(node, field, [s for s in stmts if not (isinstance(s, ast.stmt) and _is_bare_string(s))])
    return float(h_visit_ast(tree).total.effort)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def apply_floor(metrics: dict, source: str, floor: float, seed_effort: float) -> tuple[dict, str | None]:
    """The search's metrics of a program (``shinka_eval.evaluate_program``) as a simplification run
    scores it: ``combined_score`` = -Halstead effort, the floor, slack and efforts in ``public``, and
    the simplification lines before the usual feedback. Returns the metrics and None at or above the
    floor, or the reason the program is incorrect below it. Metrics without a finite score are
    returned as they are: they are a failed program, not one below the floor."""
    score = metrics.get("combined_score")
    if score is None or not math.isfinite(score):
        return metrics, None
    effort = halstead_effort(source)
    slack = score - floor
    below = slack < 0
    public = metrics.get("public", {})
    verdict = ("BELOW the floor: this program is incorrect and is not used as a parent" if below
               else "at or above the floor")
    header = [f"Simplification: validation score {score:.3f}, floor {floor:.3f}, slack {slack:+.3f} ({verdict})",
              f"Halstead effort {effort:.1f} (seed program {seed_effort:.1f}); "
              + ", ".join(_count(public[k], noun) for k, noun in (("n_params", "parameter"), ("n_latent", "latent variable"))
                          if k in public)]
    out = {**metrics, "combined_score": -effort,
           "public": {**public, "halstead_effort": round(effort, 1), "seed_effort": round(seed_effort, 1),
                      "floor": floor, "slack": round(slack, 3), "above_floor": not below},
           "private": {**metrics.get("private", {}), "halstead_effort": effort, "validation_score_x100": score},
           "text_feedback": "\n".join(header) + "\n\n" + metrics.get("text_feedback", "")}
    return out, (f"validation score {score:.3f} is below the floor {floor:.3f}" if below else None)

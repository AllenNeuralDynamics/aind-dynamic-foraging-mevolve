# Simplification runs: a hard floor on the score, Halstead effort as the objective

After the search, study 40 simplifies the search's best program at three floors (study 40 ADR 0005 and
ADR 0008). MEvolve runs each floor as its own **simplification run**: the same launch wrapper, evaluator,
islands, eval job modes, snapshots, W&B and archive as a search, with `"mode": "simplify"` in `run.json`
(decided with Han, 2026-11-08, MEvolve #83). The floors themselves, how they are computed and which
programs are simplified are study 40's decisions, not this tool's.

## The objective

- **Correct only at or above the floor.** A candidate's validation score is computed exactly as in the
  search: 2-level EM with the per-mouse fallback (ADR 0004 and its amendment), `max_em` from the config,
  ×100. The candidate is correct only if that score is ≥ `floor`. A program below the floor is incorrect
  but not lost: it stays in the database with its metrics and its feedback, and `correct.json` gives the
  reason. Shinka samples parents only from correct programs and keeps only correct ones in its archive.
- **`combined_score` = −Halstead effort** for every scored program, so Shinka, which maximizes, prefers
  the simpler of two correct programs. A program that failed to fit keeps the search's failure value,
  0.0, as Shinka itself writes for a job without results; every consumer reads `correct` first.
- **The floor is applied after the cache.** The evaluation script fits the program (or reads the repeat
  cache), then applies the floor (`simplify.apply_floor`). So the seed cache and the repeat cache hold the
  search's metrics under the search's keys, and the three floor runs can share every fit when they use
  the same `seed_cache` and `stage_dir`.

## The Halstead rule (`simplify.halstead_effort`)

1. Parse the program module's source with `ast.parse`. Comments, formatting and the evolve-block markers
   are not in the syntax tree.
2. Drop every bare string statement (docstrings included) from every statement list.
3. Take radon's total for the module: `radon.metrics.h_visit_ast(tree).total.effort`, radon 6.0.1
   (pinned in the `search` extra). Effort = difficulty × volume, difficulty = (h1 / 2) × (N2 / h2),
   volume = (N1 + N2) × log2(h1 + h2), over operators (h1 distinct, N1 in all) and operands (h2, N2).

Radon counts as operators the arithmetic, boolean, comparison and augmented-assignment operators,
unary ones included (the `-5.0` of a bound is one), and as operands what they act on. Calls, subscripts
and assignments alone add nothing, so a program heavy in `jnp.where` calls and light on arithmetic reads
as cheap. That is a property of the measure; it is kept because the number is deterministic and is
the measure study 40 ADR 0005 chose. For scale: the template scores 132, Hattori2019 397, Miller-RHG 1,414, CompareToThreshold 2,517.

## Feedback and prompt

Each program's feedback starts with its validation score, the floor, the slack (score − floor), its
Halstead effort beside the seed programs' lowest, and its numbers of parameters and latent variables,
then the search's usual generation-mice diagnostics. The task message (`prompt.simplify_task`) keeps the
search's description of the task, interface and scoring, and asks for a simpler program: fewer
parameters, latent variables and operations, plain mechanisms, at or above the floor. The interface and
the program limits are unchanged. The literature library is left out of the prompt by default.

## Seeds and islands

`seed_programs` (program files) replace the template and the literature seeds: the first is Shinka's
initial program on island 0, the others are placed on islands 1, 2, ..., as literature seeds are in a
search. All are scored before the runner starts (generation 0); a seed below the floor stops the run
before any Claude call, because its island would hold no correct program; so does a seed the program
limits reject (set `max_params` and `max_latent` to the search's). The first is scored again by
Shinka as its initial program (once per run; a shared repeat cache makes it a hit on later floors). If
that second evaluation is not correct, the run stops before any proposal (`run.simplify_runner`): when
an evaluation raises, Shinka records its initial program as correct with score 0.0, which would beat
every −effort and anchor the stopping rule where no program can pass it.

## Stopping

A simplification run stops after `stop_window_simplify` proposals (default 300) finish without a
correct program of lower effort than the best so far, or at `max_usd`. Because `combined_score` is
−effort and only correct programs count, this is the search's stopping rule (`stopping.py`) with se = 0:
the anchor starts at the lowest seed effort, a strictly lower effort resets the count, and ties,
programs below the floor and failed proposals each add one. The search's `stop_se` and `stop_window`
are refused in simplify mode.

## Config (`run.json`)

| key | default | meaning |
|---|---|---|
| `mode` | `"search"` | `"simplify"` for a simplification run |
| `floor` | none | validation score ×100 a correct program must reach; required in simplify mode |
| `seed_programs` | `[]` | program files, one island each; required in simplify mode, and `seeds` must be unset |
| `stop_window_simplify` | 300 | proposals without a lower effort before the run stops |
| `library_in_prompt` | `false` in simplify mode | as in a search |

`max_usd`, `max_em`, the model and effort, `max_params`/`max_latent` (set them to the search's, or its
programs are rejected) and the eval launch settings work as in a search. `n_proposals` still caps the
run (default 10): set it well above `stop_window_simplify`, as the search sets it above `stop_window`.
The seed program files are read again on `--resume` (Shinka's initial program and the reference
effort), so keep them on shared disk, for example in the study's variant directory.

## Considered options

- **A soft penalty, score − λ·effort.** One objective, no incorrect programs, but λ is arbitrary: any
  value sets an exchange rate between likelihood and readability that nothing in the study fixes, and
  each λ would give a different program. The floor is the pre-registered constraint (study 40 ADR 0005):
  keep 90%, 75% or 50% of the gain, then be as simple as possible. A hard floor states exactly that.
- **A separate simplification loop outside Shinka.** It would duplicate the islands, archive, dollar
  cap, snapshots and W&B logging that the search already has (ADR 0001).
- **Halstead difficulty or plain counts** (parameters, latent variables, lines): study 40 ADR 0005
  chose Halstead effort; the counts are reported beside it.

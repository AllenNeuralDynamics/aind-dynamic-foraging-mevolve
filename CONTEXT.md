# MEvolve

The vocabulary of the tool that evolves cognitive programs of mouse dynamic foraging. Hierarchy and
evaluation terms (subject, session, held-out subject, context session, condition) are the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md)'s
and keep that meaning here; the study's own terms are in study 40's `CONTEXT.md`.

## Language

### Programs

**Program**:
A short JAX module that predicts each choice from the session's earlier choices and rewards, with
named parameters; the unit the search evolves.
_Avoid_: model (that names a model family), agent

**Parameter**:
A named scalar a program declares with bounds, fitted to data by the evaluator and never chosen by
the LLM.
_Avoid_: weight

**Latent variable**:
A continuous quantity a program carries from trial to trial, such as a Q value; reset at the start
of each session.
_Avoid_: state, hidden state

**State**:
A discrete hidden state, as in a GLM-HMM (for example engaged or disengaged). A program with
states carries its belief over them as latent variables.
_Avoid_: mode, regime

**Session facts**:
Known properties of a session given to a program at its start: training stage, task variant and
the session's index in the subject's history.
_Avoid_: metadata, covariates

**Seed program**:
A program that starts a search, such as a published model rewritten in program form.
_Avoid_: initial program

**Seed library**:
The published models of `aind-dynamic-foraging-models` rewritten as programs, each checked against
its original.

### Fitting and scoring

**Session level**:
The fitting regime in which each session's parameters vary around the subject's typical values,
with no population level.
_Avoid_: hierarchical model, random effects

**Typical values**:
The parameter values of a subject around which its sessions vary.
_Avoid_: subject mean

**Session marginal**:
A held-out session's score averaged over candidate parameter sets, each weighted by how well it
explained the session's earlier trials.

**Cohort**:
A named set of subjects the evaluator fits and scores (generation, validation, selection, test).
Which subjects belong to each is a study decision.

**Replay**:
Running a program through a real session's choices and rewards to get per-trial predictions.
_Avoid_: closed loop (in `aind-dynamic-foraging-models` that name means replay)

**Rollout**:
Running a fitted program as an agent in the task simulator, so it generates its own choices.
_Avoid_: closed loop, simulation

**Replay fingerprint**:
A program's predicted choice statistic on real sessions set against the subjects' actual one after the
same history, such as the switch probability after k consecutive unrewarded trials.
_Avoid_: aux stat, summary statistic

**Rollout fingerprint**:
A behavioural statistic of rolled-out sessions set against the same statistic of real sessions, such as
the switch curve by reward and run length.
_Avoid_: generative match

### Search

**Proposal**:
One LLM call that returns a new program and the hypothesis behind it.
_Avoid_: sample, mutation

**Evaluation**:
Fitting and scoring one program on the generation and validation cohorts.
_Avoid_: run, trial

**Score**:
A program's pooled held-out likelihood per trial on the validation cohort, reported ×100.
_Avoid_: fitness, reward

**Feedback**:
The diagnostic text the LLM sees about a program, such as where it loses by stage and task variant.

**Elite**:
A program kept in the archive for its score.

**Island**:
A sub-population of programs evolved apart from the others, with occasional migration.

**Search run**:
One evolution: one ShinkaEvolve run, one launch, one W&B run.
_Avoid_: experiment, sweep

**Gate**:
A check that must pass before the next build step or any spending.

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
A program that starts a search, such as a literature model rewritten in program form.
_Avoid_: initial program

**Literature model**:
A cognitive model from the literature, as implemented in `aind-dynamic-foraging-models` and fitted per
subject by study 12, which defines the term.
_Avoid_: published model, author model

**Seed library**:
The literature models study 12 ranks, rewritten as programs, each checked against its original.

### Fitting and scoring

**Fitting regimes** are named by their levels and their method (decided with Han, 2026-11-02):

| label | levels (parameters vary around…) | method | where |
|---|---|---|---|
| **per session** | none: each session fitted alone | maximum likelihood | the lab's per-session fitting, [aind-analysis-wrapper-df-model-fitting](https://github.com/AllenNeuralDynamics/aind-analysis-wrapper-df-model-fitting) |
| **per subject** | none: one parameter set per subject | maximum likelihood | study 12 r1 (regime 1) |
| **3-level HB** | session → subject → population | full posterior (NUTS) | study 08, which did not mix |
| **2-level HB** | session → subject | full posterior (NUTS) | study 13 (HB-CTT, HB-Hattori) |
| **2-level EM** | session → subject | Laplace-EM | MEvolve's fitter; study 12 r3 (EM-CTT, ...) |

**2-level model**:
The model in which each session's parameters vary around the subject's typical values, with no
population level. MEvolve fits it by Laplace-EM (2-level EM); study 13 samples its full posterior
(2-level HB).
_Avoid_: session level (reads as each session fitted alone), hierarchical model (2 or 3 levels?),
random effects, EM alone (GLM-HMMs are also fitted by EM)

**Fitter**:
The code that fits a program to a subject's sessions under one fitting regime (per subject or 2-level
EM) and returns its parameters (typical values and spreads, for 2-level EM) and its latent variables on
every trial. Fitting and reading out latent variables is a use in its own right, not only a step of a
search; the fitter moves to `aind-dynamic-foraging-models` once gate 4 passes (ADR 0005).
_Avoid_: evaluator (that is the search's use of the fitter)

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

**Evaluator**:
The search's use of the fitter: fit a program on each subject's fit sessions, score its held-out
sessions, and return the score and the feedback.
_Avoid_: fitter

**Evaluation**:
One run of the evaluator on one program over the generation and validation cohorts.
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

**Simplification run**:
A search run that rewrites given programs to be simpler: a candidate counts only if its score is at or
above the run's floor, and among those a lower Halstead effort is better (ADR 0006).
_Avoid_: pruning, distillation

**Floor**:
The validation score a simplification run's programs must keep, one per run; study 40 sets it.
_Avoid_: threshold (that names a literature model's mechanism)

**Halstead effort**:
A program's size and difficulty in one number, from its operators and operands; docstrings and comments
do not count (ADR 0006 gives the exact rule).
_Avoid_: complexity (too broad)

**Gate**:
A check that must pass before the next build step or any spending.

### Scientist mode

**Scientist mode**:
MEvolve's third mode (`"mode": "scientist"`, ADR 0007): a Claude agent forms hypotheses about the mice,
tests them with MEvolve's tools, and confirms registered claims once on mice not used to find them.
Search mode, which optimises one score, does not change.
_Avoid_: MEvolve 2.0 (it is not a newer search), harness alone (search mode has one too)

**Scientist agent**:
The Claude agent that runs scientist mode's loop. Its own term because *agent* alone is on the
_Avoid_ list of **program**: a program is never an agent here.
_Avoid_: agent alone, optimiser

**Reviewer**:
A second agent, started by the scientist agent with a fresh context, that tries to break a claim before
it may be confirmed; only it can record a verdict.
_Avoid_: critic, judge

**Episode**:
One session of the scientist agent with a fresh context, opened from the lab notebook.
_Avoid_: run (a run holds many episodes), generation

**Lab notebook**:
The run's files: notes, tool results, claims, reviews, looks, budget and episode summaries. It, not the
agent's context, carries the work from episode to episode.
_Avoid_: memory, log

**Tier**:
A set of mice with a rule for who may read it: **discovery** (study 40's four cohorts; read freely
through the tools), **confirmation** (the rest of the training pool; read only inside `confirm`) and
**test** (the held-out pool; no tool reads it). The using study commits the lists.
_Avoid_: cohort (a cohort is one named subject set inside a tier), split

**Claim**:
A registered statement about the mice with a model check and a model-free check, each a statistic with
a direction and a criterion; immutable once registered (its id is the sha256 of its contents).
_Avoid_: finding (a claim becomes a finding only if it passes its look), hypothesis (not yet registered)

**Look**:
One run of a claim's checks on all confirmation mice, once; every look is recorded, failures too, and a
study has a fixed number of them.
_Avoid_: test (that names the test tier), validation

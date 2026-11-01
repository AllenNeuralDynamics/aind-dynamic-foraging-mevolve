# AGENTS.md — aind-dynamic-foraging-mevolve

MEvolve is the search tool for study 40 of `aind-dynamic-foraging-bfm-studies`. This file holds
the rules an agent must follow here. The vocabulary is in [`CONTEXT.md`](CONTEXT.md) and the
decisions with their reasons are in [`docs/adr/`](docs/adr/); read both before changing the design.

## Explaining and reporting style

- Plain language, about 80% of ASD-STE100: short sentences, active voice, no filler.
- Explain like a teacher: the first time a specialised term appears, say what it means and why it
  is used here.
- Prefer a figure to a table and a table to prose.
- Times are Seattle time (`TZ=America/Los_Angeles`).

## Workflow

- Every tracked change starts with an issue assigned to an owner, on project 184
  (`AIND-behavior-fm`) with Status, Priority and Size set. File first, then work.
- Open a PR for every change; use Conventional Commits; never squash-merge; never merge without
  Han's explicit confirmation. Request a Copilot review on code and rules PRs and resolve every
  thread.
- Refer to issues in other repos as `AllenNeuralDynamics/<repo>#N`; a bare `#N` means this repo.
- Write code test-first. The reference implementations in `aind-dynamic-foraging-models` are the
  oracle for parity tests.

## Science lives in study 40, not here

- Results, figures, frozen inputs and reports belong to study 40 in the studies repo and follow
  its rules. Never commit run outputs here.
- Never retype another study's number. Comparator values are read from the studies repo's
  committed JSON.

## Compute

- Never run Python on the HPC login node (`hpc-code`). Use `srun --ntasks=1` or `sbatch` on a
  compute node; plain `srun` on partition `aind` runs the job several times.
- Use the `mevolve` conda env. Do not install into the shared `dynamic-foraging-bfm-cpu` env.

## Secrets

- API keys live only in `~/.config/mevolve/` (`wandb_api_key`, `anthropic_api_key`; mode 600).
  MEvolve's jobs export them as `WANDB_API_KEY` and `ANTHROPIC_API_KEY` themselves, so `~/.netrc`
  keeps the personal W&B key for the rest of the stack.
- Never put a key in this repo, an sbatch file, a log, a commit message or a chat.
- W&B: entity `NeuralDynamics`, project `aind-dynamic-foraging-model`.

## Coding principles

Adapted from Karpathy's observations on LLM coding pitfalls, as in the studies repo:

1. **Think before coding.** State assumptions; surface trade-offs; ask when unclear.
2. **Simplicity first.** The minimum code that solves the problem; nothing speculative.
3. **Surgical changes.** Touch only what the task needs; clean up only your own mess.
4. **Goal-driven execution.** Turn each task into a check that proves it worked.

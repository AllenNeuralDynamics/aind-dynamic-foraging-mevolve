# ShinkaEvolve runs the search, pinned and shimmed

The search runs on [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI,
Apache-2.0), pinned to one commit, with a small compatibility shim in this repo, instead of a loop
we write ourselves. For round 1 the evaluator holds the science and most of the risk, and it is ours
either way; Shinka already brings islands, an archive, novelty filtering, meta-recommendations,
asynchronous evaluation, a dollar cap, W&B logging and a web UI with a program genealogy, which
saves about one to one and a half weeks (decided 2026-10-31 after reading Shinka's source).

## Considered options

- **Our own loop.** Full control and a natural place for research on the search itself, at about one
  to one and a half weeks more work. Kept as the fallback if the engine smoke test fails.
- **OpenEvolve.** Reaches Claude only through the OpenAI-compatibility layer (no prompt caching,
  effort ignored) or the Claude Code CLI.

## Consequences

- The shim fixes what we found in Shinka's source: it sends a thinking budget and a temperature that
  Claude 5.x rejects, sends no prompt caching, bills models missing from its price table at $0
  (which would silently disable the cap), and treats refusals as ordinary replies.
- Shinka prints a program's score with two decimals, so the evaluator reports the likelihood ×100.
- The launch wrapper imports Shinka's runner in-process, so the shim applies; shelling out to
  `shinka_run` would bypass it.
- Shinka runs with its local launcher inside one SLURM job; its SLURM launcher submits one job per
  candidate and is not used.
- Research on the search itself (a Pareto archive, tempered islands, tree search) means forking
  Shinka or replacing it.

# The fitter moves to aind-dynamic-foraging-models once gate 4 passes

MEvolve's code that fits programs (per subject and 2-level EM, MEvolve ADR 0004) is a **fitter**, not
only part of the search's **evaluator**: it returns each program's fitted parameters and its latent
variables on every trial, which later work reads directly, for example to relate latent variables to
photometry and Neuropixels recordings. The lab's other fitting methods already live in
`aind-dynamic-foraging-models`: per-session and per-subject MLE (each forager's `fit`) and 2-level and
3-level HB (`hierarchical_bayes/`). 2-level EM landed in MEvolve only because it grew inside the
evaluator during gate 1c, with no decision about where it belongs (decided with Han, 2026-11-02).

So, in order:

1. Now: the code is split into a fitter (fit, return parameters and latent variables) and an evaluator
   (fit, score held-out sessions, return score and feedback), and the glossary separates the two.
2. Once gate 4 (recovery of simulated subjects) passes, so the fitter has passed both parity and
   recovery and its interface has settled: the fitter and the program interface it fits move to
   `aind-dynamic-foraging-models`, beside `hierarchical_bayes/`. MEvolve then depends on a pinned
   commit of the library and keeps only the evaluator, the search and the gates.
3. No one uses the fitter's latent variables for physiology before the move.

Results produced before the move (gates 1b and 1c, study 12 r3) keep the MEvolve commit they ran on;
nothing is rerun because of the move.

## Considered options

- **Rename only, the fitter stays in MEvolve.** Cheapest, but the lab's fitting methods stay split
  across two repos, and anyone who wants latent variables has to depend on a search tool.
- **Move now.** Would stall the evaluator's command line (MEvolve #20) and study 12's r3 launch by
  days, while the cost study for the search is still changing the fitter's search-time settings; the
  move would likely have to be done twice.

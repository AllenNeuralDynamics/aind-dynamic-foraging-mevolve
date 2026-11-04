# Runs log to the NeuralDynamics W&B account with their own key

MEvolve logs to W&B entity `NeuralDynamics`, project `aind-dynamic-foraging-model`, not to the
`AIND-disRNN` entity the rest of the stack uses, because Han's personal account has a 200 GB
storage cap and the NeuralDynamics account does not. Its key is a different W&B user from the
personal one, so it lives in its own file, `~/.config/mevolve/wandb_api_key`, and is exported as
`WANDB_API_KEY` only by MEvolve's jobs; `~/.netrc` can hold one key per host and keeps the personal
key for everything else (decided with Han, 2026-10-31).

## Amendment (2026-11-03): project `dfbfm-mevolve`

New runs log to project `dfbfm-mevolve` in the same entity, with the same key. Under the studies
repo's naming rule (studies issue 99), a harness gets its own project, `dfbfm-<name>`, and nothing new
goes into the shared `aind-dynamic-foraging-model` (decided by Han, 2026-11-03). Runs already
logged there stay there and are not copied, and a freeze or report that reads them keeps reading them
from that project.

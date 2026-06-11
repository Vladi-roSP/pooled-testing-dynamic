# compute_center/ — heavy runs for the cluster (do NOT run on a laptop)

These scripts wrap the existing CLIs with the large parameters meant for a
compute center. They are intentionally **not executed locally** (RL at scale and
the large-N MILP/Gurobi sweeps take many CPU-hours). Everything they call is the
*corrected* code on branch `augmented/correctness-and-paper`.

All output goes under `results/cluster/` (created on first run).

## 1. `run_rl_scale.sh` — PPO bucket policies at scale
Trains the `DaptsBucketEnv` PPO policy (no MOSEK dependency) at N=50 for the
grid B∈{2,3,4}, G∈{3,5}, 3 seeds, 200k timesteps each (the docstring's target;
the only models in `rl_models/` today are 8192-step smoke checks). Each run also
evaluates against the augmented myopic greedy on matched episodes and writes a
CSV. Estimated cost: ~18 runs × (200k steps) — hours on CPU, minutes on GPU.

```bash
bash augmented/compute_center/run_rl_scale.sh
```

## 2. `run_heavy_sweeps.sh` — large-N optimal-vs-greedy sweeps
Runs `overnight_experiments.py` for n∈{20,30,50} (needs a Gurobi or MOSEK
license; falls back to the heuristic pool otherwise — see the import fix in
`pool_solvers.py`) and the `sprint3_experiments.py` main battery at the full
instance counts. These populate the large-scale rows the paper references for
"does the augmented benefit grow with scale".

```bash
bash augmented/compute_center/run_heavy_sweeps.sh
```

## Reproducibility
Seeds are fixed in each script. The augmented package is deterministic given a
seed (gym `np_random` + PPO seed). Re-running with the same seed reproduces the
numbers. Record the git commit (`git rev-parse HEAD`) alongside the outputs.

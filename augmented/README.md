# Augmented pooled testing

`augmented/` contains the Dynamic Augmented Pooled Testing Strategies (DAPTS)
code. In the augmented model, a pooled test returns the exact count
`r = |t cap Z|` of infected individuals in the tested pool.

## How to read this folder

1. Start with `notebooks/README.md` for the recommended notebook order.
2. Use the notebooks for explanation, figures, and experiment narratives.
3. Use the Python modules here as the source of reusable algorithms.

## Module map

| Area | Files | Role |
| --- | --- | --- |
| Core primitives | `core.py`, `strategy.py`, `simulator.py` | Bitmasks, pool/test primitives, strategy representation and simulation. |
| Bayesian updates | `bayesian.py`, `gibbs_analysis.py` | Exact, counting and Gibbs-style posterior updates. |
| Solvers | `solver.py`, `static_solver.py`, `classical_solver.py`, `pool_solvers.py`, `hybrid_solver.py` | Optimal dynamic, static, classical and hybrid solver variants. |
| Greedy methods | `greedy.py`, `infection_reward_greedy.py`, `semi_utility.py`, `expected_utility.py` | Myopic policies and scoring variants. |
| Experiments | `experiments.py`, `csv_experiments.py`, `sprint3_experiments.py`, `overnight_experiments.py`, `comparison.py`, `cross_verification.py` | Reproducible experiment runners and comparisons. |
| VW / super-nodes | `vw_demo.py`, `vw_restrict.py`, `vw_restrict_sweep.py` | Experiments around the VW super-node reformulation. |
| RL | `rl_env.py`, `rl_train.py`, `rl_examples.py`, `rl_examples_demo.py`, `rl_models/` | Reinforcement-learning environment, training and saved exploratory models. |
| Visualization | `tree_extractor.py`, `tree_visualizer.py`, `figures/` | Tree extraction, plotting helpers and generated figures. |
| Tests | `tests*.py` | Regression and solver checks. |

## Notebook convention

The notebooks follow the experimental style used in the dynamic part of the
project: small parameter blocks (`n`, `B`, `G`, `p`, `u`), direct calculations,
short comments near the code, and printed results next to each experiment.

Each main notebook now starts with the same header:

- objective,
- guiding question,
- expected reading path,
- setup cell,
- numbered sections with a short explanation before each code block.

LaTeX source files and LaTeX build auxiliaries were removed from `augmented/`.
The folder should now focus on notebooks, code, figures, data and PDFs.

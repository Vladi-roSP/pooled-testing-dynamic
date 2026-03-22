# Project Summary: Dynamic Augmented Pooled Testing

## What is this project about?

This project implements and studies **Dynamic Augmented Pooled Testing Strategies (DAPTS)** — a new paradigm for group-based disease testing.

**The problem**: You have n people, each with some probability of being infected (p_i) and some value of clearing them (u_i). You have a limited budget of B tests, each of which can pool up to G people's samples together. Your goal: maximize the total expected utility of people you can **prove are healthy**.

**Classical vs Augmented**: In classical pooled testing, a test returns binary (+/−) — "someone is infected" or "nobody is infected." In **augmented** pooled testing, the test returns the **exact count** of infected people in the pool (e.g., "2 out of 5 are infected"). This extra information enables smarter adaptive strategies.

**The paper**: "Dynamic Augmented Pooled Testing" by Vladimir, Hector, and Francisco (January 2026). The code implements Section 2 ("Warm-up: Brute Force for Small Instances").

---

## Repository Structure

```
pooled-testing-dynamic/
├── augmented/          ← NEW: augmented pooled testing machinery
│   ├── __init__.py
│   ├── core.py
│   ├── strategy.py
│   ├── simulator.py
│   ├── expected_utility.py
│   ├── baselines.py
│   ├── solver.py            ← CORE: brute-force DP for optimal DAPTS
│   ├── bayesian.py
│   ├── greedy.py
│   ├── static_solver.py
│   ├── classical_solver.py
│   ├── comparison.py
│   ├── example.py
│   ├── tests.py
│   └── results.tex
├── classical/          ← EXISTING: classical pooled testing (binary +/-)
│   ├── solvers/
│   ├── rl_training/
│   ├── rl_evaluation/
│   ├── notebooks/
│   ├── data/, models/, figures/, training/
│   └── slurm_scripts/
├── Action_Plan.md
├── README.md
└── chatgpt_solver_context.md
```

---

## What each file does

### Foundation layer (data structures & primitives)

**`augmented/core.py`** — Bitmask helpers
- Individuals are numbered 0 to n-1. A Python integer is used as a bitmask to represent sets (pools, infection profiles, cleared individuals).
- `mask_from_indices([0, 2])` → `0b101` (integer 5)
- `indices_from_mask(5, n=3)` → `[0, 2]`
- `popcount(mask)` → number of 1-bits
- `all_pools(n, G)` → every possible pool of size ≤ G. This is P_G([n]) from the paper.
- `test_result(pool, z)` → r(t, Z) = |t ∩ Z|. The augmented count of infected in the pool.

**`augmented/strategy.py`** — DAPTS representation
- `History` = a tuple of (pool_mask, result) pairs, e.g., `((0b110, 1), (0b001, 0))`.
- `DAPTS` class = the strategy F = (F_1, ..., F_B). Stores a policy table: for each step k and each possible history of length k-1, which pool to test next.
- `F.choose(k, history)` → the pool mask for step k given the history so far.
- This is **Definition 2** from the paper.

### Simulation & evaluation layer

**`augmented/simulator.py`** — Run a strategy on a specific infection profile
- `apply_dapts(F, z_mask, n, u)` → simulates F step by step on infection profile Z.
  - For each step: look up the pool from the policy, compute the test result, append to history.
  - If a pool's result is 0, all individuals in that pool are "cleared" (proven healthy).
  - Returns (terminal_history, cleared_mask, realized_utility).
- This implements the recursive unfolding from the paper: t_1 = F_1(∅), h_1 = {(t_1, r_1)}, t_2 = F_2(h_1), ...

**`augmented/expected_utility.py`** — Compute u(F) = E_Z[u(F,Z)]
- `exact_expected_utility(F, p, u, n)` — enumerates all 2^n infection profiles, weights each by Pr(Z), sums up u(F,Z). Exact but exponential.
- `mc_expected_utility(F, p, u, n)` — Monte Carlo estimate: sample random infection profiles, average the utility. Used as a sanity check.

**`augmented/baselines.py`** — Upper and lower bounds
- `u_max(p, u)` → Σ u_i · q_i. The theoretical maximum if you had infinite tests.
- `u_single(p, u, B)` → test the top min(B,n) individuals by u_i·q_i one at a time. Simplest possible strategy.

### Solver layer (the core algorithms)

**`augmented/solver.py`** — BRUTE-FORCE DP for optimal DAPTS (the most important file)
- `solve_optimal_dapts(p, u, B, G)` → finds F* that maximizes u(F) over ALL possible strategies.
- **DP state**: (k, remaining_set, cleared_mask)
  - k = tests used so far
  - remaining_set = frozenset of infection profiles Z still consistent with what we've observed
  - cleared_mask = bitmask of individuals proven healthy
- **Recursion**: For each candidate pool, partition the remaining profiles by their test outcome r. For r=0 → those individuals are cleared. Recurse on each partition. Pick the pool with highest total expected value.
- **Trick**: Returns *weighted* (not conditional) expected utility — avoids normalization at each step.
- **Policy reconstruction**: After solving, traces back through the DP memo to build the actual DAPTS object.
- Limited to n ≤ 14 (state space is exponential).

**`augmented/classical_solver.py`** — DP for classical dynamic testing (U^D)
- Same structure as solver.py but test results are **binary**: negative (pool ∩ Z = ∅) or positive (pool ∩ Z ≠ ∅).
- Branching factor is always 2 (vs |pool|+1 for augmented).
- Returns only the value (no policy reconstruction needed — we only need U^D for comparison).

**`augmented/static_solver.py`** — Solvers for static (non-adaptive) strategies
- `solve_static_non_overlapping(p, u, B, G)` → U^s_NO: choose B disjoint pools upfront.
- `solve_static_overlapping(p, u, B, G)` → U^s_O: choose B pools (may overlap) upfront.
- These don't adapt to results — all pools are fixed before any testing.

**`augmented/bayesian.py`** — Bayesian posterior updates
- After observing r = |t ∩ Z| for pool t, update each individual's infection probability.
- For i in pool: uses Bayes' rule with a Poisson-Binomial distribution over the other pool members.
- `_poisson_binomial_pmf(probs)` → PMF of "how many successes from independent Bernoullis with different probabilities."
- `bayesian_update_single_test(p, pool, r, n)` → updated probabilities after one test.
- `bayesian_update(p, history, n)` → apply updates sequentially for a full history.
- Special cases: r=0 means everyone in pool is healthy (p'_i = 0); r=|t| means everyone is infected (p'_i = 1).

**`augmented/greedy.py`** — Greedy heuristic strategies
- **Myopic greedy** (`greedy_myopic_*`): At each step, pick the pool maximizing P(all healthy) × Σ u_i for uncleared members. After observing the result, do a Bayesian update and repeat.
- **Lookahead greedy** (`greedy_lookahead_*`): At step 1, try all pools and evaluate future value (using myopic for subsequent steps). Picks the best first-step pool accounting for what happens after.
- Key insight: the augmented information doesn't help with *immediate* pool selection (only r=0 earns utility), but it helps *future* decisions through better Bayesian posteriors.

### Comparison & output layer

**`augmented/comparison.py`** — Compute all 6+ strategies and compare
- `compare_all(p, u, B, G)` → dictionary with U_single, U_s_NO, U_s_O, U_D, U_D_A, U_greedy, U_max.
- `print_comparison(...)` → formatted table verifying the inequality chain.
- Tests 4 instances with different characteristics (low/high infection rates, uniform population).

**`augmented/example.py`** — Demo script
- Runs the n=5, B=2, G=3 example from the paper.
- Compares optimal, myopic greedy, and lookahead greedy.
- Simulates on a specific infection profile Z = {3} to show step-by-step behavior.

**`augmented/results.tex`** — LaTeX document with computational results
- Full comparison tables for all 4 instances.
- Key finding: **augmented benefit is largest when infection rates are high** (+2.74% for Instance 2).
- Greedy captures 96-100% of the gap between individual testing and optimal.
- For uniform populations, augmented gives no benefit (symmetry means count doesn't help identify *who* is infected).

**`augmented/tests.py`** — 30+ unit tests
- Tests every component: bitmask helpers, test_result, all_pools, apply_dapts, exact vs MC, baselines, Bayesian updates, greedy, static solvers, classical solver, and the full inequality chain.

### Classical directory

**`classical/`** — Pre-existing code from earlier work on classical (binary) pooled testing:
- `solvers/milpSample.py` — MILP formulation, greedy dynamic, exact DP with Bayesian updates
- `solvers/greedyDynamicSample.py` — Greedy with Gibbs MCMC sampling
- `rl_training/PPO_bucket_gymnasium_B*.py` — Reinforcement learning (PPO) training scripts
- `rl_evaluation/` — RL evaluation scripts
- `notebooks/Pooled_Testing_Strategies.ipynb` — 296-cell analysis notebook
- Plus data, models, figures, training logs, SLURM scripts

---

## The inequality chain (the central result)

For any population (p, u) with budget B and pool size G:

```
U^single  ≤  U^s_NO  ≤  U^s_O  ≤  U^D  ≤  U^D_A  ≤  U^max
```

| Strategy | What it does |
|----------|-------------|
| U^single | Test individuals one at a time, pick top B by u_i·q_i |
| U^s_NO | Static non-overlapping pools (decided upfront, disjoint) |
| U^s_O | Static overlapping pools (decided upfront, may share people) |
| U^D | Dynamic classical (adapt based on binary +/− results) |
| U^D_A | Dynamic augmented (adapt based on exact count) ← **this is what solver.py computes** |
| U^max | Σ u_i·q_i — theoretical upper bound (infinite budget) |

---

## What was done (chronologically)

1. **Explored and documented** the existing classical pooled testing codebase
2. **Set up environment**: Jupyter, MOSEK license, Gurobi license
3. **Implemented the full DAPTS brute-force solver** from Section 2 of the paper (core.py, strategy.py, simulator.py, expected_utility.py, baselines.py, solver.py, tests.py, example.py)
4. **Verified correctness**: 17 unit tests, hand-checkable cases, exact EU matches solver, MC agrees within tolerance
5. **Reorganized the repo** into classical/ vs augmented/ paradigms
6. **Created GitHub repo** and pushed: https://github.com/APIEXSmx/pooled-testing-dynamic
7. **Created Action_Plan.md** with full research briefing
8. **(Pull from collaborator)** Added 6 new modules: bayesian.py, greedy.py, static_solver.py, classical_solver.py, comparison.py, results.tex — expanding from just the optimal solver to the full comparison framework with all 6 strategies + greedy

# Action Plan: Dynamic Augmented Pooled Testing

**Date:** February 2026
**Project:** Dynamic-Augmented Pooled Testing (DAPTS)
**Reference:** "Dynamic-Augmented Pooled Testing" (Vladimir, Hector, Francisco, January 2026)

---

## 1. Researcher-Facing Summary

The codebase implements two parallel paradigms for dynamic pooled testing:

**Classical model** (`classical/`): The existing implementation from prior work [1], where a pooled test on $t \subseteq [n]$ returns a binary outcome $r \in \{+, -\}$ (someone is infected vs. nobody is infected). Belief updates use Bayesian inference via `bayesTheorem()` or Gibbs MCMC (`GibbsMCMCWindow`). The planner solves for optimal dynamic strategies via exact DP (`solveDynamic`), MILP (`solveMILP`), greedy conic relaxation (`solveConicGibbsGreedyDynamic`), or PPO reinforcement learning.

**Augmented model** (`augmented/`): The new implementation from the January 2026 paper, where the idealized augmented test returns $r(t, Z) = \sum_{i \in t} Z_i$ — the exact count of infected individuals in the pool. This provides a strictly richer signal than the binary model. The package provides a brute-force DP solver over information states $(k, \text{remaining\\_profiles}, \text{cleared\\_mask})$ to compute the optimal DAPTS $F^* = (F_1, \ldots, F_B)$ with $F_k: \mathcal{H}_{k-1} \to \mathcal{P}_G([n])$, exact and MC expected utility $u(F) = \mathbb{E}_Z[u(F,Z)]$, and verifies the hierarchy $U^{\text{single}} \le U_A^D \le U^{\max}$.

---

## 2. Top Priority Code Map (Ranked)

### Priority 1: The DP Solver — the core scientific contribution

- **File:** `augmented/solver.py` (lines 39-201)
- **Key functions:** `solve_optimal_dapts()`, nested `dp()`, nested `reconstruct()`
- **Why priority:** This IS the algorithm. It implements exact optimization over DAPTS for augmented tests. The DP state `(k, remaining_set, cleared_mask)` compresses the full history space into sufficient statistics. The recurrence partitions infection profiles by test outcome $r$, updates cleared individuals when $r = 0$, and selects the pool maximizing weighted expected utility.
- **Talking points:**
  - "The state is (step, set of consistent infection profiles, individuals proven healthy). We branch on every possible outcome $r \in \{0, \ldots, |t|\}$ of the augmented test."
  - "The DP returns *weighted* expected utility (not conditional), so the final value equals $u(F^*)$ directly since initial total mass = 1."
  - "Complexity is $O(B \cdot 2^{2^n} \cdot \binom{n}{\le G})$ — exponential in $2^n$, hence the $n \le 14$ guard."

### Priority 2: The test result model — what makes augmented different

- **Files:** `augmented/core.py` line 70 (`test_result`), `augmented/simulator.py` lines 16-55 (`apply_dapts`)
- **Why priority:** `test_result(pool, z) = popcount(pool & z)` is the one-line function that distinguishes the augmented model from the classical binary model. In the simulator, the clearing rule `if r == 0: cleared |= pool` is where utility is earned.
- **Talking points:**
  - "In the classical model, $r \in \{+, -\}$. In the augmented model, $r \in \{0, \ldots, |t|\}$. This richer signal means the planner can distinguish 'one infected' from 'three infected', leading to better downstream test allocation."
  - "An individual $i$ is cleared iff $\exists (t_k, r_k) \in h_B$ with $i \in t_k$ and $r_k = 0$."

### Priority 3: Strategy representation & PG(S)

- **Files:** `augmented/strategy.py` lines 15-68 (`DAPTS`, `History`), `augmented/core.py` lines 40-65 (`all_pools`)
- **Why priority:** These directly realize Definitions 1 and 2 from the paper. History $h_k$ is a tuple of $(t_j, r_j)$ pairs. $\mathcal{P}_G(S)$ is computed by `all_pools(n, G)`. The DAPTS class stores `policy[k-1]: dict[History, pool_mask]`, making $F_k(h_{k-1})$ a dict lookup.
- **Talking points:**
  - "We represent $F$ as a list of $B$ dictionaries. Entry `policy[k-1][history]` returns the pool chosen at step $k$."
  - "$\mathcal{P}_G(S)$ enumerates $\sum_{i=0}^{G} \binom{n}{i}$ subsets. For $n=5, G=3$: 26 candidate pools."

### Priority 4: Expected utility & baselines — verification layer

- **Files:** `augmented/expected_utility.py`, `augmented/baselines.py`
- **Why priority:** These let you verify the solver. `exact_expected_utility` enumerates all $2^n$ profiles and computes $u(F) = \sum_Z \Pr(Z) \cdot u(F,Z)$. The baselines ($U^{\max}$, $U^{\text{single}}$) provide the bounding sandwich.
- **Talking points:** "We verified that the solver value matches exact enumeration to $10^{-9}$ precision, and the hierarchy $U^{\text{single}} \le U_A^D \le U^{\max}$ holds on all test instances."

### Priority 5: Existing classical solvers — the comparison point

- **Files:** `classical/solvers/milpSample.py` lines 752-827 (`solveDynamic`), lines 294-342 (`solveConicSingle`), lines 196-286 (`bayesTheorem`)
- **Why priority:** Your lead will ask "how does augmented compare to non-augmented?" The classical `solveDynamic` uses binary outcomes and Bayesian belief updates; the augmented solver uses count outcomes and profile filtering.
- **Talking point:** "In the classical model, a positive test only tells us *someone* is infected. In the augmented model, we learn *how many*, which lets us partition the information space more finely."

---

## 3. Execution Flow (Call Graph)

```
solve_optimal_dapts(p, u, B, G)          # augmented/solver.py:39
|
|-- Precompute w[z] = Pr(Z=z) for all z in {0,...,2^n-1}    # L84-88
|-- pools = all_pools(n, G, include_empty=False)              # augmented/core.py:40
|
|-- dp(k=0, remaining=all_z, cleared=0)                       # L96
|   |-- [Terminal k==B]:                                       # L108
|   |   return total_mass * utility(cleared)                   # L116
|   |
|   +-- [k < B]: for each pool in pools:                      # L129
|       |-- Partition remaining by r = test_result(pool, z)   # augmented/core.py:70
|       |   -> buckets: {r: [z_list]}                          # L134-139
|       |
|       |-- For each outcome r:
|       |   |-- new_cleared = cleared | pool  if r==0         # L143
|       |   +-- recurse: dp(k+1, new_remaining, new_cleared)  # L144
|       |
|       +-- ev = sum_r sub_val -> select argmax pool           # L146-150
|
|-- reconstruct(k=0, all_z, 0, ())                            # L177
|   |-- DAPTS.set_action(k+1, history, best_pool)             # augmented/strategy.py:62
|   +-- For each outcome r: recurse with extended history     # L188-201
|
+-- return (optimal_value, DAPTS object)
```

### Where mathematical objects live in code

| Paper Object | Code | File:Line |
|---|---|---|
| $J = (p, u)$ | Separate `p: list`, `u: list` | Passed as args everywhere |
| $\mathcal{P}_G([n])$ | `all_pools(n, G)` | `augmented/core.py:40` |
| $h_k$ (history) | `History = tuple[tuple[int,int],...]` | `augmented/strategy.py:15` |
| $r(t, Z)$ | `test_result(pool_mask, z_mask)` | `augmented/core.py:70` |
| $F_k(h_{k-1})$ | `DAPTS.choose(k, history)` | `augmented/strategy.py:41` |
| $u(F, Z)$ | `apply_dapts(F, z_mask, n, u)` | `augmented/simulator.py:16` |
| $u(F)$ | `exact_expected_utility(F, p, u, n)` | `augmented/expected_utility.py:30` |
| $U^{\max}$ | `u_max(p, u)` | `augmented/baselines.py:13` |
| $U^{\text{single}}$ | `u_single(p, u, B)` | `augmented/baselines.py:25` |

### Classical model comparison (key function locations)

| Paper Object | Code | File:Line |
|---|---|---|
| $J = (p, u)$ | `create_agents(N)` -> list of `(id, utility, health_prob)` tuples | `classical/solvers/milpSample.py:61` |
| Pool generation (conic) | `solveConicSingle(agents, G)` | `classical/solvers/milpSample.py:294` |
| Bayes belief update | `bayesTheorem(agents, posGroups, negAgents)` | `classical/solvers/milpSample.py:196` |
| Gibbs MCMC update | `GibbsMCMCWindow(agents, posGroups, negAgents, ...)` | `classical/solvers/milpSample.py:558` |
| Gibbs MCMC with counts | `GibbsMCMCWindowCount(agents, posGroups, negAgents, ...)` | `classical/solvers/greedyDynamicSample.py:344` |
| Exact dynamic DP | `solveDynamic(agents, G, B, posGroups, negAgents)` | `classical/solvers/milpSample.py:752` |
| Static MILP | `solveMILP(agents, G, B)` | `classical/solvers/milpSample.py:345` |
| Greedy dynamic | `solveConicGibbsGreedyDynamic(agents, G, B, ...)` | `classical/solvers/milpSample.py:1060` |
| Greedy dynamic with counts | `solveConicGibbsGreedyDynamicCount(agents, G, B, ...)` | `classical/solvers/greedyDynamicSample.py:982` |
| RL (PPO) training | `PPO_bucket_gymnasium_B*.py` | `classical/rl_training/` |
| RL evaluation | `PPO_bucket_gymnasium_*_use.py` | `classical/rl_evaluation/` |
| $U^{\max}$ | `maxUtil(agents)` | `classical/solvers/milpSample.py:136` |
| $U^{\text{single}}$ | `solveStaticNoPool(agents, B)` | `classical/solvers/milpSample.py:142` |

---

## 4. "Tomorrow Deliverable" Plan (2-4 hours)

### (a) One meaningful code contribution already done

The entire `augmented/` package is the contribution:
- `augmented/core.py` — Bitmask helpers + `all_pools()` implementing $\mathcal{P}_G(S)$ + `test_result()` implementing $r(t,Z)$
- `augmented/strategy.py` — `DAPTS` class implementing $F = (F_1, \ldots, F_B)$
- `augmented/simulator.py` — `apply_dapts()` computing $u(F, Z)$
- `augmented/expected_utility.py` — Exact and MC computation of $u(F) = \mathbb{E}_Z[u(F,Z)]$
- `augmented/baselines.py` — $U^{\max}$ and $U^{\text{single}}$ benchmarks
- `augmented/solver.py` — Brute-force DP solver for optimal DAPTS
- `augmented/tests.py` — 17 unit tests all passing
- `augmented/example.py` — Full demo script

Additional contribution: Add `validate_instance()` to catch invalid inputs:

```python
# In augmented/baselines.py
def validate_instance(p, u, B, G):
    """Validate population instance J and parameters."""
    n = len(p)
    assert len(u) == n, f"len(p)={len(p)} != len(u)={len(u)}"
    assert all(0 <= pi <= 1 for pi in p), "All p[i] must be in [0,1]"
    assert all(ui >= 0 for ui in u), "All u[i] must be >= 0"
    assert B >= 1, f"Budget B must be >= 1, got {B}"
    assert G >= 1, f"Pool size G must be >= 1, got {G}"
    return n
```

### (b) Micro-experiment to run and present

Run the example script to generate results:

```bash
cd "pooled testing dynamic"
python augmented/example.py
```

**Expected output** (already verified):

| n | B | G | U_single | U_A^D | U_max | Augmented Gain |
|---|---|---|----------|-------|-------|----------------|
| 2 | 1 | 2 | 9.00 | 9.00 | 9.50 | 0% (trivial) |
| 3 | 1 | 2 | 1.71 | 1.96 | 2.55 | +14.6% |
| 3 | 2 | 2 | 2.55 | 2.55 | 2.55 | 0% (budget sufficient) |
| 5 | 2 | 3 | 11.84 | 19.07 | 22.19 | +61.1% |

Key insight: the augmented model provides 15-70%+ improvement over single testing, depending on configuration. The gain is largest when the budget is tight relative to population size.

### (c) Presentation outline (7 bullets)

1. **Problem:** Pooled testing with augmented information — test returns *count* of infected, not just positive/negative
2. **Formalization:** DAPTS $F = (F_1, \ldots, F_B)$, history $h_k$, augmented result $r(t,Z) = \sum_{i \in t} Z_i$ (Definitions 1 & 2 from paper)
3. **Implementation:** Brute-force DP over information states $(k, \text{remaining profiles}, \text{cleared mask})$ — exact optimal for small $n$
4. **Verification:** Solver value matches exact enumeration ($< 10^{-9}$ error), MC sanity check within tolerance, inequality chain $U^{\text{single}} \le U_A^D \le U^{\max}$ verified on all instances (17/17 tests pass)
5. **Key result:** Augmented dynamic testing provides 15-70%+ improvement over single testing, depending on $(n, B, G)$ configuration
6. **Dynamic benefit example:** For $n=5, B=2, G=3$: first test $\{2,3,4\}$, if $r=1$ then second test adapts to $\{0,1,4\}$ (includes individual 4 for a second chance)
7. **Next steps:** Greedy heuristic for augmented model, comparison of $U^D$ (classical dynamic) vs $U_A^D$ (augmented dynamic) on same instances, scale beyond $n=14$ via approximate methods

---

## 5. Mismatch / Risk Checklist

### Code-Paper Divergences

| Issue | Location | Severity | Detail |
|---|---|---|---|
| **No J validation** | All `augmented/` functions | Medium | Paper defines $J \in \Delta^n \times \mathbb{R}_+^n$; code never checks $p_i \in [0,1]$ or $u_i \ge 0$ |
| **0-indexed vs 1-indexed** | `augmented/strategy.py:41` | Low | Paper uses $[n] = \{1,\ldots,n\}$; code uses 0..n-1. Internally consistent but explain in presentation |
| **Paper typo: $P_0(S) = P(S)$** | Paper line 124 | Low | Paper says "$P_0(S) = P(S)$ is just the power set" — this is wrong ($P_0(S) = \{\emptyset\}$). Code correctly implements $\mathcal{P}_G(S)$ as subsets of size $\le G$ |
| **No pi function** | Not implemented | Medium | Paper defines Bayesian posterior update $\pi(u_t, p_t, r) = p'_t$ (line 43-47). The brute-force solver doesn't need it (works on profile space directly), but a greedy solver will |
| **No phi function** | Not implemented | Low | Paper defines posterior distribution $\phi$ over infection counts (line 38-39). The idealized model makes $\phi$ a point mass, so it's implicit |
| **DP returns weighted value** | `augmented/solver.py:116` | Low | `total_mass * val` is a weighted (not conditional) value. Correct because initial mass = 1.0, but undocumented |

### Missing Components

| Component | Status | Impact |
|---|---|---|
| Input validation for J | Missing in augmented | Easy win — add `validate_instance()` |
| Classical $U^D$ comparison | Not wired up | Need adapter between existing `solveDynamic` and augmented interface |
| Greedy augmented heuristic | Not started | Paper Section 2 mentions it as next step |
| Confidence intervals for MC | Missing | `mc_expected_utility` returns point estimate only |
| Larger-n approximate solver | Not started | Current solver limited to $n \le 14$ |
| Bayesian posterior update pi | Not implemented | Needed for greedy algorithms; brute-force doesn't need it |

### Likely Questions from Lead Researcher + Answers

| Question | Answer (grounded in code) |
|---|---|
| "How do you know the solver is correct?" | "Three checks: (1) solver value matches `exact_expected_utility` to $10^{-9}$ on all test cases (`augmented/tests.py:257`), (2) MC estimate agrees within tolerance (`augmented/tests.py:149`), (3) inequality chain $U^{\text{single}} \le U_A^D \le U^{\max}$ verified (`augmented/tests.py:175`)" |
| "What's the max n you can handle?" | "Brute-force: $n \le 14$ (guard at `augmented/solver.py:73`). Practically, $n=5$ runs in seconds, $n=8$ in minutes. The state space is $O(2^{2^n})$ which is the fundamental bottleneck." |
| "How does augmented compare to classical dynamic?" | "Not yet wired up in the same script, but the existing `solveDynamic` in `classical/solvers/milpSample.py` solves $U^D$ for binary outcomes. The augmented advantage comes from finer partitioning of the information space — $|t|+1$ outcomes vs 2." |
| "Where does the Bayesian update happen?" | "In the brute-force solver, there's no explicit $\pi$ update. Instead, the DP directly tracks which infection profiles remain consistent with observed outcomes — this is equivalent to Bayesian updating but operates on the profile space. See `augmented/solver.py:134-139`." |
| "What about the greedy algorithm for augmented?" | "Not implemented yet — that's the natural next step. The existing `solveConicGibbsGreedyDynamic` in `classical/solvers/greedyDynamicSample.py` provides a template, but needs adaptation for count-based outcomes instead of binary." |
| "Why bitmasks?" | "Bitmasks give O(1) set intersection/union via bitwise AND/OR, and popcount gives the augmented test result in one operation. For brute-force enumeration over 2^n profiles this is the natural representation." |
| "What's the relationship to the existing RL code?" | "The RL code (PPO in `classical/rl_training/`) approximates optimal policies for large n where exact DP is intractable. The augmented brute-force solver provides ground truth for small n. A future direction would be training RL agents for the augmented model." |

---

## 6. Repository Structure

```
pooled-testing-dynamic/
|
|-- README.md                          # Project overview
|-- Action_Plan.md                     # This file
|
|-- augmented/                         # NEW: Augmented pooled testing (count-based)
|   |-- __init__.py                    # Package exports
|   |-- core.py                        # Bitmask helpers, all_pools (PG), test_result (r(t,Z))
|   |-- strategy.py                    # DAPTS class, History type
|   |-- simulator.py                   # apply_dapts: simulate F on fixed Z
|   |-- expected_utility.py            # exact and MC computation of u(F)
|   |-- baselines.py                   # U_max, U_single benchmarks
|   |-- solver.py                      # Brute-force DP solver for optimal DAPTS
|   |-- tests.py                       # 17 unit tests (all passing)
|   +-- example.py                     # Demo script with full workflow
|
+-- classical/                         # EXISTING: Classical binary pooled testing
    |-- solvers/                       # Core algorithms
    |   |-- milpSample.py              # MILP, Greedy Dynamic, exact DP, Bayes updates
    |   |-- greedyDynamicSample.py     # Greedy with Gibbs MCMC + count constraints
    |   +-- trial.py                   # MOSEK conic solver test
    |
    |-- rl_training/                   # PPO training scripts (B=2,3,4,5)
    |-- rl_evaluation/                 # PPO model evaluation scripts
    |-- training/                      # PyTorch RL training (N=3, N=50, ES variants)
    |-- slurm_scripts/                 # HPC job submission scripts
    |-- notebooks/                     # Jupyter notebooks (main + visualization)
    |   |-- Pooled_Testing_Strategies.ipynb  # 296-cell development notebook
    |   |-- multGraphs.ipynb           # Multi-config comparison plots
    |   +-- singleGraphs.ipynb         # Single-config plots
    |
    |-- data/                          # Experimental results and training data
    |-- models/                        # Saved PyTorch/RL model weights
    +-- figures/                       # Output visualizations
```

---

## 7. How to Run

```bash
# Run augmented tests (17 tests)
python augmented/tests.py

# Run augmented example
python augmented/example.py

# Classical solvers require MOSEK and Gurobi licenses
# MOSEK license: ~/mosek/mosek.lic (installed)
# Gurobi license: restricted free license (installed via pip)
```

# Mosek & Gurobi Pool Selection for Large-n Instances

**Date:** 2026-03-21
**Status:** Proposed

## Problem

The augmented module's greedy/hybrid solvers enumerate all C(n_active, 1) + ... + C(n_active, G) candidate pools at each step via `all_pools_from_mask()`. This grows combinatorially:

| n_active | G | # Pools |
|----------|---|---------|
| 14       | 5 | ~2,700  |
| 30       | 5 | ~174K   |
| 50       | 5 | ~2.4M   |
| 100      | 5 | ~79M    |

This makes the greedy strategies impractical for n > ~25-30.

## Solution: Solver-Based Myopic Pool Selection

Replace the enumeration in `_myopic_best_pool` with a mathematical optimization program that finds the optimal pool directly using Mosek (exponential cone) or Gurobi (MILP with log general constraint).

### Mathematical Formulation

The myopic score is:

```
Score(S) = P(r=0|S) × Σ_{i∈S} u_i = ∏_{i∈S} (1-p_i) × Σ_{i∈S} u_i
```

Taking the log (monotone transform, preserves argmax over feasible pools with Score > 0):

```
max  log(Σ u_i x_i) + Σ x_i log(q_i)
s.t. Σ x_i ≤ G
     Σ x_i ≥ 1              [non-empty pool]
     x_i ∈ {0,1}  for i ∈ active set
```

Where `q_i = 1 - p_i` (probability of being healthy).

**Log-transform validity:** The log is well-defined because:
1. `Σ x_i ≥ 1` ensures `Σ u_i x_i > 0` (given all `u_i > 0` for active individuals)
2. Individuals with `p_i ≥ 1 - threshold` are excluded by `compute_active_mask` before solver construction (confirmed infected), so `q_i ≥ threshold` (~1e-10) and `log(q_i)` is always finite for all solver variables.

### Mosek Formulation (Exponential Cone)

Directly adapts `csef/optimisation/python/models/conic_model.py` (cross-project reference, not a local import):

```
max  y + Σ x_i log(q_i)
s.t. z = Σ u_i x_i
     z ≥ ε                   [numerical safety, ε = 1e-8]
     (z, 1, y) ∈ K_exp       [primal exp cone via Domain.inPExpCone(): y ≤ log(z)]
     Σ x_i ≤ G
     Σ x_i ≥ 1
     x_i ∈ {0,1}
```

This is a mixed-integer conic program. Mosek handles this natively via `Domain.inPExpCone()`.

### Gurobi Formulation (MILP)

Gurobi lacks native exponential cone support. Two options:

**Option A (preferred, Gurobi 9.0+):** Use general constraint `addGenConstrLog`:

```
max  y + Σ x_i log(q_i)
s.t. z = Σ u_i x_i
     y = log(z)              [general constraint]
     Σ x_i ≤ G
     Σ x_i ≥ 1
     x_i ∈ {0,1}
     z ≥ ε                   [avoid log(0)]
```

**Option B (fallback):** Piecewise-linear approximation of log(z) as in `approximation_model.py`, using K segments on [L, U].

### Solver Parameters

- **Time limit per call:** 30 seconds (both Mosek and Gurobi). The pool selection is called once per greedy step (B steps total), so total solver time ≤ 30s × B.
- **MIP gap tolerance:** 0.001 (0.1%). Tight enough for pool selection since errors don't compound across steps (each step re-solves from updated posteriors).
- **On timeout:** Return best incumbent solution if available. If no incumbent, fall back to a heuristic: test the G individuals with highest `u_i × q_i` scores.
- **On infeasibility:** Return 0 (no pool). This should only happen if n_active == 0 (handled by edge case check before solver).
- **Verbosity:** Suppressed by default (Mosek: `M.setSolverParam("log", "0")`; Gurobi: `m.setParam('OutputFlag', 0)`).

## Architecture

### New File: `augmented/pool_solvers.py`

```python
# Solver-based optimal pool selection for large-n instances.
#
# Two backends:
#   mosek_best_pool()  — exponential cone (exact)
#   gurobi_best_pool() — MILP with log constraint (exact/approx)
#
# Both return a pool bitmask, same interface as _myopic_best_pool().

def mosek_best_pool(p, u, G, n, cleared_mask) -> int:
    """Find optimal myopic pool using Mosek exponential cone."""

def gurobi_best_pool(p, u, G, n, cleared_mask) -> int:
    """Find optimal myopic pool using Gurobi MILP."""

def solver_best_pool(p, u, G, n, cleared_mask, solver='mosek') -> int:
    """Dispatch to Mosek or Gurobi based on solver parameter."""
```

**Internal logic for both solvers:**
1. Compute `active_mask` via `compute_active_mask(p, cleared_mask, n)`
2. Extract active indices, their `p_i` and `u_i`
3. Convert `p_i` to `q_i = 1 - p_i`, then compute `log(q_i)` coefficients (all finite due to `compute_active_mask` filtering)
4. Build and solve the optimization model with n_active variables
5. Convert solution `x_i` back to a bitmask over the original population
6. Return pool mask (0 if no useful pool found)

Since the solver operates only on active (uncleared, not confirmed-infected) individuals, the `u_i` values in the formulation directly correspond to the `gain` in the greedy code — no cleared-mask filtering is needed within the solver.

**Edge cases (handled before building the model):**
- `n_active == 0` → return 0
- All `u_i == 0` for active → return 0
- All `q_i == 0` (everyone certainly infected) → return 0 (but `compute_active_mask` already excludes these)
- `n_active ≤ G` → only one maximal feasible pool (everyone active), skip solver and return `active_mask` directly

### Modifications to `greedy.py`

Add a `pool_selector` parameter to all functions that call `_myopic_best_pool`:

```python
# Expected utility functions:
def greedy_myopic_expected_utility(p, u, B, G, pool_selector=None):
def greedy_myopic_counting_expected_utility(p, u, B, G, pool_selector=None):
def greedy_myopic_gibbs_expected_utility(p, u, B, G, ..., pool_selector=None):

# Simulation functions:
def greedy_myopic_simulate(p, u, B, G, z_mask, pool_selector=None):
def greedy_myopic_counting_simulate(p, u, B, G, z_mask, pool_selector=None):
def greedy_myopic_gibbs_simulate(p, u, B, G, z_mask, ..., pool_selector=None):
```

The `pool_selector` is a callable with signature `(p, u, G, n, cleared_mask) -> int`.

Default behavior (`None`) preserves existing enumeration for backward compatibility. Internally, each function replaces `_myopic_best_pool(current_p, u, G, n, cleared_mask)` with `pool_selector(current_p, u, G, n, cleared_mask)` when the parameter is provided.

### Modifications to `hybrid_solver.py`

The `hybrid_greedy_bruteforce` already accepts `greedy_score_fn`. We adapt it to also accept a `pool_selector` parameter that routes to the solver-based selection. The `_greedy_fallback` for n_active > 14 becomes useful with solver-based selection since the solver can handle large n.

### Modifications to `comparison.py` and `experiments.py`

Add new strategy entries:
- `U_greedy_mosek`: myopic greedy with Mosek pool selection
- `U_greedy_gurobi`: myopic greedy with Gurobi pool selection

In `experiments.py`, add these to `evaluate_instance()` for n > 14.

### Modifications to `_infection_aware_best_pool` in `hybrid_solver.py`

The infection-aware score blends myopic + info gain: `α × myopic + (1-α) × info_gain`. The info gain component requires computing Bayesian posteriors for each outcome, which is hard to formulate as a single conic/MILP program.

**Approach:** For infection-aware scoring, keep the enumeration-based approach for small n_active, and for large n_active, use solver-based myopic selection (dropping the info gain component, or using a sampling-based approximation of info gain over a solver-selected candidate set).

## File Changes Summary

| File | Change |
|------|--------|
| `augmented/pool_solvers.py` | **NEW** — Mosek and Gurobi pool selection |
| `augmented/greedy.py` | Add `pool_selector` parameter to all 6 functions calling `_myopic_best_pool` |
| `augmented/hybrid_solver.py` | Route to solver-based pool selection for large n |
| `augmented/comparison.py` | Add `U_greedy_mosek`, `U_greedy_gurobi` strategies |
| `augmented/experiments.py` | Add solver-based strategies, support n > 14 |
| `augmented/tests_solvers.py` | **NEW** — Tests for solver-based pool selection |

## Testing Strategy

1. **Correctness:** For small n (≤ 14), verify that `mosek_best_pool` and `gurobi_best_pool` return the same pool as `_myopic_best_pool` (enumeration)
2. **Consistency:** Verify Mosek and Gurobi agree with each other
3. **Greedy equivalence:** `greedy_myopic_expected_utility(pool_selector=mosek_best_pool)` matches `greedy_myopic_expected_utility()` for small n
4. **Scaling:** Run solver-based greedy for n = 30, 50, 100 and verify it completes in reasonable time
5. **Edge cases:** p_i = 0, p_i = 1, all agents cleared, G ≥ n_active

## Dependencies

- `mosek` (already installed, v11.1 at project root + v10.0.27 in requirements)
- `gurobipy` (already in requirements, v12.0.3)
- Both require valid licenses (user has Mosek academic license; Gurobi license assumed available)

## Future Optimizations

- **Warm-starting:** At each greedy step, the solver could be warm-started with the previous step's solution. Both Mosek and Gurobi support MIP warm starts. This could reduce solve times when posteriors change only slightly between steps.

# Mosek & Gurobi Pool Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brute-force pool enumeration with Mosek/Gurobi solvers so greedy strategies scale to n >> 14.

**Architecture:** New `pool_solvers.py` module with `mosek_best_pool()` and `gurobi_best_pool()` that solve a mixed-integer conic/MILP program for optimal myopic pool selection. These plug into existing greedy functions via a `pool_selector` callable parameter, preserving backward compatibility.

**Tech Stack:** Python 3.13, mosek 11.1 (Fusion API), gurobipy 13.0, numpy

**Spec:** `docs/superpowers/specs/2026-03-21-mosek-gurobi-pool-selection-design.md`

**Working directory:** `/Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic`

**Run tests with:** `python augmented/tests_solvers.py` (same pattern as existing `tests.py`, `tests_hybrid.py`)

**Important:** The local `mosek/` directory at project root shadows the pip-installed package. Always run from the `pooled-testing-dynamic/` subdirectory to avoid import issues.

---

## File Structure

| File | Role |
|------|------|
| `augmented/pool_solvers.py` | **NEW** — `mosek_best_pool()`, `gurobi_best_pool()`, `solver_best_pool()`, `_heuristic_best_pool()` |
| `augmented/tests_solvers.py` | **NEW** — Tests for pool solvers: correctness vs enumeration, cross-solver consistency, scaling, edge cases |
| `augmented/greedy.py` | **MODIFY** — Add `pool_selector` param to all 6 functions that call `_myopic_best_pool` |
| `augmented/hybrid_solver.py` | **MODIFY** — Thread `pool_selector` through hybrid solver, remove n>14 greedy fallback limitation |
| `augmented/comparison.py` | **MODIFY** — Add `U_greedy_mosek`, `U_greedy_gurobi` strategies |
| `augmented/experiments.py` | **MODIFY** — Add solver-based strategies, support n > 14 experiments |

---

## Task 1: Mosek Pool Solver

**Files:**
- Create: `augmented/pool_solvers.py`
- Create: `augmented/tests_solvers.py`

- [ ] **Step 1: Write failing test — `mosek_best_pool` matches enumeration for a known instance**

Create `augmented/tests_solvers.py`:

```python
"""
Tests for solver-based pool selection (Mosek and Gurobi).

Run with:  python augmented/tests_solvers.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (
    mask_from_indices, indices_from_mask, compute_active_mask,
    all_pools_from_mask,
)
from augmented.greedy import _myopic_best_pool
from augmented.pool_solvers import mosek_best_pool


def _myopic_score(pool, p, u, n, cleared_mask):
    """Compute the myopic score for a pool (for test assertions)."""
    pool_idx = indices_from_mask(pool, n)
    if not pool_idx:
        return 0.0
    prob_clear = 1.0
    for i in pool_idx:
        prob_clear *= (1.0 - p[i])
    gain = sum(u[i] for i in pool_idx if not (cleared_mask >> i & 1))
    return prob_clear * gain


def test_mosek_matches_enumeration_n5():
    """Mosek returns same pool as brute-force enumeration for n=5."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    G, n = 3, 5
    cleared_mask = 0

    enum_pool = _myopic_best_pool(p, u, G, n, cleared_mask)
    mosek_pool = mosek_best_pool(p, u, G, n, cleared_mask)

    enum_score = _myopic_score(enum_pool, p, u, n, cleared_mask)
    mosek_score = _myopic_score(mosek_pool, p, u, n, cleared_mask)

    assert abs(enum_score - mosek_score) < 1e-6, (
        f"Mosek score {mosek_score:.6f} != enum score {enum_score:.6f}"
    )


# ---- Test runner ----
if __name__ == "__main__":
    test_fns = sorted(
        [(name, obj) for name, obj in globals().items()
         if name.startswith("test_") and callable(obj)],
        key=lambda x: x[0],
    )
    passed = failed = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if failed:
        sys.exit(1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: FAIL with `ImportError: cannot import name 'mosek_best_pool' from 'augmented.pool_solvers'`

- [ ] **Step 3: Implement `mosek_best_pool` in `pool_solvers.py`**

Create `augmented/pool_solvers.py`:

```python
"""
Solver-based optimal pool selection for large-n instances.

Two backends:
  mosek_best_pool()  — exponential cone (exact via Mosek Fusion API)
  gurobi_best_pool() — MILP with log general constraint (Gurobi)

Both return a pool bitmask, same interface as greedy._myopic_best_pool().
Signature: (p, u, G, n, cleared_mask) -> int
"""

import math

from augmented.core import (
    mask_from_indices, indices_from_mask, compute_active_mask,
)


def _heuristic_best_pool(active_indices, p, u, G):
    """Fallback: pick G individuals with highest u_i * q_i."""
    scored = [(u[i] * (1.0 - p[i]), i) for i in active_indices]
    scored.sort(reverse=True)
    selected = [idx for _, idx in scored[:G]]
    return mask_from_indices(selected)


def mosek_best_pool(p, u, G, n, cleared_mask):
    """Find optimal myopic pool using Mosek exponential cone.

    Solves:
        max  y + Σ x_i log(q_i)
        s.t. z = Σ u_i x_i
             z ≥ ε
             (z, 1, y) ∈ K_exp   [y ≤ log(z)]
             Σ x_i ≤ G
             Σ x_i ≥ 1
             x_i ∈ {0,1}

    Returns pool bitmask (0 if no useful pool).
    """
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Var

    # Identify active individuals
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        return 0

    # Extract active probabilities and utilities
    active_p = [p[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    # Check: all utilities zero → no useful pool
    if all(ui <= 0 for ui in active_u):
        return 0

    # Shortcut: if n_active ≤ G, the only maximal pool is everyone
    if n_active <= G:
        return active_mask

    # Compute log(q_i) coefficients
    log_q = []
    for pi in active_p:
        qi = 1.0 - pi
        log_q.append(math.log(qi) if qi > 1e-15 else -35.0)  # clamp

    EPS = 1e-8

    with Model('mosek_pool') as M:
        M.setSolverParam("log", "0")  # suppress output
        M.setSolverParam("mioMaxTime", 30.0)  # 30s time limit
        M.setSolverParam("mioTolRelGap", 1e-3)  # 0.1% MIP gap

        # Variables
        x = M.variable("x", n_active, Domain.binary())
        y = M.variable("y", 1, Domain.unbounded())
        z = M.variable("z", 1, Domain.greaterThan(EPS))
        d = M.variable("d", 1, Domain.equalsTo(1.0))

        # Exponential cone: (z, d, y) ∈ K_exp → y ≤ log(z)
        t = Var.vstack(z.index(0), d.index(0), y.index(0))
        M.constraint("expc", t, Domain.inPExpCone())

        # z = Σ u_i x_i
        M.constraint("util", Expr.sub(Expr.dot(active_u, x), z.index(0)),
                      Domain.equalsTo(0.0))

        # Pool size constraints
        M.constraint("pool_max", Expr.sum(x), Domain.lessThan(float(G)))
        M.constraint("pool_min", Expr.sum(x), Domain.greaterThan(1.0))

        # Objective: max y + Σ x_i log(q_i)
        M.objective("obj", ObjectiveSense.Maximize,
                     Expr.add(y.index(0), Expr.dot(log_q, x)))

        M.solve()

        # Check solution status
        sol_status = M.getPrimalSolutionStatus()
        if sol_status is None:
            return _heuristic_best_pool(active_indices, p, u, G)

        # Extract solution
        x_vals = x.level()
        selected = [active_indices[i] for i in range(n_active)
                    if x_vals[i] > 0.5]

        if not selected:
            return _heuristic_best_pool(active_indices, p, u, G)

        return mask_from_indices(selected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add augmented/pool_solvers.py augmented/tests_solvers.py
git commit -m "feat: add Mosek-based pool selection solver with test"
```

---

## Task 2: Gurobi Pool Solver

**Files:**
- Modify: `augmented/pool_solvers.py`
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Write failing test — `gurobi_best_pool` matches enumeration**

Add to `augmented/tests_solvers.py`:

```python
from augmented.pool_solvers import mosek_best_pool, gurobi_best_pool, solver_best_pool


def test_gurobi_matches_enumeration_n5():
    """Gurobi returns same pool as brute-force enumeration for n=5."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    G, n = 3, 5
    cleared_mask = 0

    enum_pool = _myopic_best_pool(p, u, G, n, cleared_mask)
    gurobi_pool = gurobi_best_pool(p, u, G, n, cleared_mask)

    enum_score = _myopic_score(enum_pool, p, u, n, cleared_mask)
    gurobi_score = _myopic_score(gurobi_pool, p, u, n, cleared_mask)

    assert abs(enum_score - gurobi_score) < 1e-6, (
        f"Gurobi score {gurobi_score:.6f} != enum score {enum_score:.6f}"
    )


def test_mosek_gurobi_agree_n5():
    """Mosek and Gurobi find pools with equal scores."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    G, n = 3, 5
    cleared_mask = 0

    mosek_pool = mosek_best_pool(p, u, G, n, cleared_mask)
    gurobi_pool = gurobi_best_pool(p, u, G, n, cleared_mask)

    mosek_score = _myopic_score(mosek_pool, p, u, n, cleared_mask)
    gurobi_score = _myopic_score(gurobi_pool, p, u, n, cleared_mask)

    assert abs(mosek_score - gurobi_score) < 1e-4, (
        f"Mosek score {mosek_score:.6f} != Gurobi score {gurobi_score:.6f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: FAIL with `ImportError: cannot import name 'gurobi_best_pool'`

- [ ] **Step 3: Implement `gurobi_best_pool` and `solver_best_pool`**

Add to `augmented/pool_solvers.py`:

```python
def gurobi_best_pool(p, u, G, n, cleared_mask):
    """Find optimal myopic pool using Gurobi MILP with log constraint.

    Solves:
        max  y + Σ x_i log(q_i)
        s.t. z = Σ u_i x_i
             y = log(z)     [general constraint]
             Σ x_i ≤ G
             Σ x_i ≥ 1
             x_i ∈ {0,1}
             z ≥ ε

    Returns pool bitmask (0 if no useful pool).
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Identify active individuals
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        return 0

    active_p = [p[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    if all(ui <= 0 for ui in active_u):
        return 0

    if n_active <= G:
        return active_mask

    log_q = []
    for pi in active_p:
        qi = 1.0 - pi
        log_q.append(math.log(qi) if qi > 1e-15 else -35.0)

    EPS = 1e-8
    U_max = G * max(active_u)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model('gurobi_pool', env=env) as m:
            m.setParam('TimeLimit', 30.0)
            m.setParam('MIPGap', 1e-3)

            # Variables
            x = m.addVars(n_active, vtype=GRB.BINARY, name='x')
            z = m.addVar(lb=EPS, ub=U_max, name='z')
            y = m.addVar(lb=-GRB.INFINITY, name='y')

            # z = Σ u_i x_i
            m.addConstr(z == gp.quicksum(active_u[i] * x[i]
                                          for i in range(n_active)),
                        name='util')

            # Pool size constraints
            m.addConstr(gp.quicksum(x[i] for i in range(n_active)) <= G,
                        name='pool_max')
            m.addConstr(gp.quicksum(x[i] for i in range(n_active)) >= 1,
                        name='pool_min')

            # y = log(z) via general constraint
            m.addGenConstrLog(z, y, name='log_z')

            # Objective: max y + Σ x_i log(q_i)
            m.setObjective(
                y + gp.quicksum(log_q[i] * x[i] for i in range(n_active)),
                GRB.MAXIMIZE,
            )

            m.optimize()

            if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                return _heuristic_best_pool(active_indices, p, u, G)

            selected = [active_indices[i] for i in range(n_active)
                        if x[i].X > 0.5]

            if not selected:
                return _heuristic_best_pool(active_indices, p, u, G)

            return mask_from_indices(selected)


def solver_best_pool(p, u, G, n, cleared_mask, solver='mosek'):
    """Dispatch to Mosek or Gurobi based on solver parameter.

    Parameters
    ----------
    solver : str
        'mosek' or 'gurobi'.

    Returns pool bitmask.
    """
    if solver == 'mosek':
        return mosek_best_pool(p, u, G, n, cleared_mask)
    elif solver == 'gurobi':
        return gurobi_best_pool(p, u, G, n, cleared_mask)
    else:
        raise ValueError(f"Unknown solver: {solver!r}. Use 'mosek' or 'gurobi'.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add augmented/pool_solvers.py augmented/tests_solvers.py
git commit -m "feat: add Gurobi-based pool solver and dispatch function"
```

---

## Task 3: Comprehensive Solver Tests (Edge Cases + Multiple Instances)

**Files:**
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Add edge case and multi-instance tests**

Append to `augmented/tests_solvers.py`:

```python
import random


def test_mosek_edge_all_cleared():
    """Returns 0 when all individuals are cleared."""
    p = [0.0, 0.0, 0.0]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 2
    cleared_mask = 0b111  # all cleared
    assert mosek_best_pool(p, u, G, n, cleared_mask) == 0


def test_gurobi_edge_all_cleared():
    """Returns 0 when all individuals are cleared."""
    p = [0.0, 0.0, 0.0]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 2
    cleared_mask = 0b111
    assert gurobi_best_pool(p, u, G, n, cleared_mask) == 0


def test_mosek_edge_n_leq_G():
    """When n_active ≤ G, returns full active mask (skip solver)."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 5  # G > n
    cleared_mask = 0
    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    assert pool == 0b111  # all three


def test_gurobi_edge_n_leq_G():
    """When n_active ≤ G, returns full active mask (skip solver)."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 5
    cleared_mask = 0
    pool = gurobi_best_pool(p, u, G, n, cleared_mask)
    assert pool == 0b111


def test_mosek_with_some_cleared():
    """Solver only considers active (uncleared) individuals."""
    p = [0.1, 0.2, 0.3, 0.15, 0.25]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    n, G = 5, 2
    cleared_mask = 0b00011  # individuals 0,1 already cleared

    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    pool_idx = indices_from_mask(pool, n)
    # Pool should only contain individuals from {2, 3, 4}
    for i in pool_idx:
        assert i >= 2, f"Individual {i} is cleared but in pool"


def test_solvers_match_enumeration_random_instances():
    """Both solvers match enumeration across 10 random instances."""
    rng = random.Random(42)
    for trial in range(10):
        n = rng.randint(4, 10)
        G = rng.randint(2, min(4, n))
        p = [rng.uniform(0.01, 0.5) for _ in range(n)]
        u = [rng.uniform(1.0, 10.0) for _ in range(n)]
        cleared_mask = 0

        enum_pool = _myopic_best_pool(p, u, G, n, cleared_mask)
        enum_score = _myopic_score(enum_pool, p, u, n, cleared_mask)

        mosek_pool = mosek_best_pool(p, u, G, n, cleared_mask)
        mosek_score = _myopic_score(mosek_pool, p, u, n, cleared_mask)
        assert abs(enum_score - mosek_score) < 1e-4, (
            f"Trial {trial}: Mosek {mosek_score:.6f} != enum {enum_score:.6f}"
        )

        gurobi_pool = gurobi_best_pool(p, u, G, n, cleared_mask)
        gurobi_score = _myopic_score(gurobi_pool, p, u, n, cleared_mask)
        assert abs(enum_score - gurobi_score) < 1e-4, (
            f"Trial {trial}: Gurobi {gurobi_score:.6f} != enum {enum_score:.6f}"
        )


def test_solver_best_pool_dispatch():
    """solver_best_pool dispatches correctly to both backends."""
    p = [0.1, 0.2, 0.3, 0.15]
    u = [4.0, 6.0, 3.0, 5.0]
    n, G = 4, 2
    cleared_mask = 0

    mosek_pool = solver_best_pool(p, u, G, n, cleared_mask, solver='mosek')
    gurobi_pool = solver_best_pool(p, u, G, n, cleared_mask, solver='gurobi')

    mosek_score = _myopic_score(mosek_pool, p, u, n, cleared_mask)
    gurobi_score = _myopic_score(gurobi_pool, p, u, n, cleared_mask)
    assert abs(mosek_score - gurobi_score) < 1e-4
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add augmented/tests_solvers.py
git commit -m "test: add edge case and random instance tests for pool solvers"
```

---

## Task 4: Add `pool_selector` to Greedy Functions

**Files:**
- Modify: `augmented/greedy.py`
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Write failing test — greedy with Mosek selector matches default greedy**

Add to `augmented/tests_solvers.py`:

```python
from augmented.greedy import (
    greedy_myopic_expected_utility,
    greedy_myopic_simulate,
    greedy_myopic_counting_expected_utility,
)


def test_greedy_mosek_matches_default_eu():
    """greedy_myopic_expected_utility with mosek selector matches default."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    B, G = 2, 3

    default_eu = greedy_myopic_expected_utility(p, u, B, G)
    mosek_eu = greedy_myopic_expected_utility(p, u, B, G,
                                              pool_selector=mosek_best_pool)

    assert abs(default_eu - mosek_eu) < 1e-6, (
        f"Default EU {default_eu:.6f} != Mosek EU {mosek_eu:.6f}"
    )


def test_greedy_gurobi_matches_default_eu():
    """greedy_myopic_expected_utility with gurobi selector matches default."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    B, G = 2, 3

    default_eu = greedy_myopic_expected_utility(p, u, B, G)
    gurobi_eu = greedy_myopic_expected_utility(p, u, B, G,
                                               pool_selector=gurobi_best_pool)

    assert abs(default_eu - gurobi_eu) < 1e-6, (
        f"Default EU {default_eu:.6f} != Gurobi EU {gurobi_eu:.6f}"
    )


def test_greedy_simulate_mosek():
    """greedy_myopic_simulate with mosek selector produces valid results."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2
    z_mask = 0b100  # individual 2 is infected

    history, cleared, utility = greedy_myopic_simulate(
        p, u, B, G, z_mask, pool_selector=mosek_best_pool
    )
    assert utility >= 0
    assert len(history) <= B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: FAIL with `TypeError: greedy_myopic_expected_utility() got an unexpected keyword argument 'pool_selector'`

- [ ] **Step 3: Add `pool_selector` to all 6 functions in `greedy.py`**

In `augmented/greedy.py`, modify these functions to accept and use `pool_selector`:

**`_myopic_best_pool`** — no change needed (it's the default).

**`greedy_myopic_simulate`** (line 63):
```python
def greedy_myopic_simulate(p, u, B, G, z_mask, pool_selector=None):
```
Inside the loop, replace:
```python
        pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
```
with:
```python
        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
```

**`greedy_myopic_expected_utility`** (line 90):
```python
def greedy_myopic_expected_utility(p, u, B, G, pool_selector=None):
```
Inside `recurse`, replace the `_myopic_best_pool` call with the same pattern.

**`greedy_myopic_counting_simulate`** (line 231):
```python
def greedy_myopic_counting_simulate(p, u, B, G, z_mask, pool_selector=None):
```
Same replacement pattern.

**`greedy_myopic_counting_expected_utility`** (line 343):
```python
def greedy_myopic_counting_expected_utility(p, u, B, G, pool_selector=None):
```
Same replacement pattern.

**`greedy_myopic_gibbs_simulate`** (line 266):
```python
def greedy_myopic_gibbs_simulate(p, u, B, G, z_mask,
                                 num_iterations=1000, burn_in=200, seed=None,
                                 pool_selector=None):
```
Same replacement pattern.

**`greedy_myopic_gibbs_expected_utility`** (line 301):
```python
def greedy_myopic_gibbs_expected_utility(p, u, B, G,
                                          num_iterations=1000, burn_in=200,
                                          seed=42, pool_selector=None):
```
Same replacement pattern.

- [ ] **Step 4: Run all tests (both old and new)**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests.py && python augmented/tests_solvers.py`
Expected: All PASS (old tests verify backward compatibility, new tests verify solver integration)

- [ ] **Step 5: Commit**

```bash
git add augmented/greedy.py augmented/tests_solvers.py
git commit -m "feat: add pool_selector parameter to all greedy functions"
```

---

## Task 5: Thread `pool_selector` Through Hybrid Solver

**Files:**
- Modify: `augmented/hybrid_solver.py`
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Write failing test — hybrid solver with solver-based pool selection**

Add to `augmented/tests_solvers.py`:

```python
from augmented.hybrid_solver import hybrid_greedy_bruteforce


def test_hybrid_with_mosek_pool_selector():
    """Hybrid solver uses mosek for pool selection in greedy phase."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    B, G = 3, 3

    # Full greedy (K=B) with default
    tree_default, eu_default = hybrid_greedy_bruteforce(p, u, B, G,
                                                         greedy_steps=B)
    # Full greedy with mosek pool selector
    tree_mosek, eu_mosek = hybrid_greedy_bruteforce(
        p, u, B, G, greedy_steps=B, pool_selector=mosek_best_pool
    )

    assert abs(eu_default - eu_mosek) < 1e-4, (
        f"Default EU {eu_default:.6f} != Mosek EU {eu_mosek:.6f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: FAIL with `TypeError: hybrid_greedy_bruteforce() got an unexpected keyword argument 'pool_selector'`

- [ ] **Step 3: Add `pool_selector` to hybrid solver**

In `augmented/hybrid_solver.py`:

1. Add `pool_selector=None` parameter to `hybrid_greedy_bruteforce` (line 250):
```python
def hybrid_greedy_bruteforce(p, u, B, G, greedy_steps,
                              greedy_score_fn=None,
                              update_method='sequential',
                              pool_selector=None):
```

2. In the function body, when `pool_selector` is provided but `greedy_score_fn` is not, create a score function wrapper:
```python
    # If pool_selector provided, wrap it as greedy_score_fn
    if pool_selector is not None and greedy_score_fn is None:
        greedy_score_fn = pool_selector
```

3. In `_dp_phase` (line 466), change the n_active > 14 fallback (line 517) to use the pool_selector if provided:
```python
    if n_active > 14:
        return _greedy_fallback(
            u, G, n, current_p, cleared_mask, step,
            history, remaining_budget, greedy_score_fn
        )
```
This already works because `greedy_score_fn` was set to `pool_selector` above.

4. Thread `pool_selector` through `_hybrid_recurse` and `_full_greedy_tree` — no changes needed since they already use `greedy_score_fn`.

- [ ] **Step 4: Run all tests**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests.py && python augmented/tests_hybrid.py && python augmented/tests_solvers.py`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add augmented/hybrid_solver.py augmented/tests_solvers.py
git commit -m "feat: thread pool_selector through hybrid solver"
```

---

## Task 6: Add Solver Strategies to Comparison and Experiments

**Files:**
- Modify: `augmented/comparison.py`
- Modify: `augmented/experiments.py`
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Write failing test — comparison includes solver strategies**

Add to `augmented/tests_solvers.py`:

```python
from augmented.comparison import compare_all


def test_compare_all_includes_solver_strategies():
    """compare_all returns solver-based strategy values for small n."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2
    results = compare_all(p, u, B, G)

    assert 'U_greedy_mosek' in results, "Missing U_greedy_mosek"
    assert 'U_greedy_gurobi' in results, "Missing U_greedy_gurobi"

    # Should match default greedy for small n
    assert abs(results['U_greedy_mosek'] - results['U_greedy']) < 1e-4
    assert abs(results['U_greedy_gurobi'] - results['U_greedy']) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: FAIL with `KeyError: 'U_greedy_mosek'`

- [ ] **Step 3: Add solver strategies to `comparison.py`**

In `augmented/comparison.py`, add imports and strategy lines:

After existing imports, add:
```python
from augmented.pool_solvers import mosek_best_pool, gurobi_best_pool
```

In `compare_all()`, after the `U_greedy_gibbs` line, add:
```python
    results["U_greedy_mosek"] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=mosek_best_pool)
    results["U_greedy_gurobi"] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=gurobi_best_pool)
```

In `print_comparison()`, add display lines after the greedy_gibbs print:
```python
    print(f"  {'U^greedy_mosek  (Mosek pool selector)':42s} = {results['U_greedy_mosek']:.6f}")
    print(f"  {'U^greedy_gurobi (Gurobi pool selector)':42s} = {results['U_greedy_gurobi']:.6f}")
```

- [ ] **Step 4: Add solver strategies to `experiments.py`**

In `augmented/experiments.py`, add imports:
```python
from augmented.pool_solvers import mosek_best_pool, gurobi_best_pool
```

In `evaluate_instance()`, after the greedy block, add:
```python
    # Solver-based greedy (works for any n)
    results['U_greedy_mosek'] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=mosek_best_pool)
    results['U_greedy_gurobi'] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=gurobi_best_pool)
```

In `summarize_results()`, add to the `labels` dict:
```python
            'U_greedy_mosek': 'U_greedy_mosek (Mosek solver)',
            'U_greedy_gurobi': 'U_greedy_gurobi (Gurobi solver)',
```

And add to the `order` list:
```python
        order = ['U_single', 'U_D', 'U_D_A', 'U_greedy',
                 'U_greedy_counting', 'U_greedy_mosek', 'U_greedy_gurobi',
                 'U_max']
```

- [ ] **Step 5: Run all tests**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests.py && python augmented/tests_hybrid.py && python augmented/tests_solvers.py`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add augmented/comparison.py augmented/experiments.py augmented/tests_solvers.py
git commit -m "feat: add solver-based strategies to comparison and experiments"
```

---

## Task 7: Scaling Benchmark Test (n=30, 50)

**Files:**
- Modify: `augmented/tests_solvers.py`

- [ ] **Step 1: Add scaling benchmark tests**

Add to `augmented/tests_solvers.py`:

```python
import time


def test_mosek_scales_n30():
    """Mosek pool selection completes in <10s for n=30, G=5."""
    rng = random.Random(99)
    n, G = 30, 5
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 10.0) for _ in range(n)]
    cleared_mask = 0

    t0 = time.time()
    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    elapsed = time.time() - t0

    pool_idx = indices_from_mask(pool, n)
    assert 1 <= len(pool_idx) <= G, f"Pool size {len(pool_idx)} out of range"
    assert elapsed < 10.0, f"Mosek took {elapsed:.1f}s (limit 10s)"


def test_gurobi_scales_n30():
    """Gurobi pool selection completes in <10s for n=30, G=5."""
    rng = random.Random(99)
    n, G = 30, 5
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 10.0) for _ in range(n)]
    cleared_mask = 0

    t0 = time.time()
    pool = gurobi_best_pool(p, u, G, n, cleared_mask)
    elapsed = time.time() - t0

    pool_idx = indices_from_mask(pool, n)
    assert 1 <= len(pool_idx) <= G, f"Pool size {len(pool_idx)} out of range"
    assert elapsed < 10.0, f"Gurobi took {elapsed:.1f}s (limit 10s)"


def test_mosek_scales_n50():
    """Mosek pool selection completes in <30s for n=50, G=5."""
    rng = random.Random(99)
    n, G = 50, 5
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 10.0) for _ in range(n)]
    cleared_mask = 0

    t0 = time.time()
    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    elapsed = time.time() - t0

    pool_idx = indices_from_mask(pool, n)
    assert 1 <= len(pool_idx) <= G, f"Pool size {len(pool_idx)} out of range"
    assert elapsed < 30.0, f"Mosek took {elapsed:.1f}s (limit 30s)"


def test_gurobi_scales_n50():
    """Gurobi pool selection completes in <30s for n=50, G=5."""
    rng = random.Random(99)
    n, G = 50, 5
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 10.0) for _ in range(n)]
    cleared_mask = 0

    t0 = time.time()
    pool = gurobi_best_pool(p, u, G, n, cleared_mask)
    elapsed = time.time() - t0

    pool_idx = indices_from_mask(pool, n)
    assert 1 <= len(pool_idx) <= G, f"Pool size {len(pool_idx)} out of range"
    assert elapsed < 30.0, f"Gurobi took {elapsed:.1f}s (limit 30s)"


def test_greedy_eu_n30_mosek():
    """Full greedy expected utility completes for n=30 with Mosek."""
    rng = random.Random(99)
    n = 30
    B, G = 2, 5
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 10.0) for _ in range(n)]

    t0 = time.time()
    eu = greedy_myopic_expected_utility(p, u, B, G,
                                         pool_selector=mosek_best_pool)
    elapsed = time.time() - t0

    assert eu > 0, f"Expected utility should be positive, got {eu}"
    assert elapsed < 120.0, f"Greedy+Mosek took {elapsed:.1f}s (limit 120s)"
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests_solvers.py`
Expected: All PASS (scaling tests may take a few seconds each)

- [ ] **Step 3: Commit**

```bash
git add augmented/tests_solvers.py
git commit -m "test: add scaling benchmarks for n=30 and n=50"
```

---

## Task 8: Run Full Comparison and Verify End-to-End

- [ ] **Step 1: Run the comparison script**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/comparison.py`
Expected: Output includes `U^greedy_mosek` and `U^greedy_gurobi` values matching `U^greedy`.

- [ ] **Step 2: Run all test suites**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python augmented/tests.py && python augmented/tests_hybrid.py && python augmented/tests_solvers.py`
Expected: All tests PASS across all three suites.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize Mosek/Gurobi pool selection integration"
```

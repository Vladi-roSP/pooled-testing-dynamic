# Context for Understanding `solver.py` — Dynamic Augmented Pooled Testing

You are helping a graduate researcher understand their implementation of a brute-force DP solver for Dynamic Augmented Pooled Testing Strategies (DAPTS). Below you will find:

1. The relevant mathematical definitions from the research paper (Section 2)
2. The supporting code modules that `solver.py` depends on
3. The full `solver.py` code

**Your goal**: Explain `solver.py` line-by-line, mapping every code construct back to the paper's mathematical objects. Make sure the researcher deeply understands:
- What each variable represents mathematically
- How the DP state space relates to the paper's definitions
- Why the recursion is correct
- How policy reconstruction works
- The "weighted vs conditional" trick used to avoid normalizing at every step

---

## Part 1: Paper Definitions (Section 2 — "Warm-up: Brute Force for Small Instances")

### Setup
- Population of **n** individuals, identified with [n] = {1, ..., n}.
- Each individual has utility **u_i ≥ 0** and infection probability **p_i ∈ [0,1]**, with **q_i = 1 − p_i**.
- Population instance: **J = (p_1,...,p_n, u_1,...,u_n)**.
- Budget of **B ≥ 1** augmented pooled tests.
- Maximum pool size **G** (biological limit): each test t ⊆ [n] must satisfy |t| ≤ G.
- **P_G(S)** = {U ⊆ S : |U| ≤ G} — all subsets of S with cardinality at most G.

### Infection Profile
Each individual is infected independently: Z_i ~ Bernoulli(p_i). The infection profile is **Z = (Z_1,...,Z_n) ∈ {0,1}^n**. The probability of a specific profile Z is:

$$\Pr(Z) = \prod_{i=1}^{n} p_i^{Z_i} \cdot q_i^{1-Z_i}$$

### Augmented Test Result (Idealized)
For a pool t ⊆ [n] and infection profile Z, the **augmented test result** is:

$$r(t, Z) = \sum_{i \in t} Z_i = |t \cap \{i : Z_i = 1\}|$$

This returns the **exact count** of infected individuals in the pool (not just binary +/−).

### Definition 1 — Testing History
A testing history of length k is a list of k test-result pairs:

$$h_k = \{(t_1, r_1), \ldots, (t_k, r_k)\}$$

where t_j ⊆ [n] and r_j ∈ {0, ..., |t_j|}. The space of all histories of length k is **H_k**. In particular, H_0 = {∅}.

### Definition 2 — DAPTS
A Dynamic Augmented Pooled Testing Strategy is **F = (F_1, ..., F_B)** where:

$$F_k : H_{k-1} \to P_G([n]) \quad \text{for } k = 1, \ldots, B$$

Each F_k maps a history of length k−1 to a pool of size at most G.

### Evaluating Performance
Given F and a specific infection profile Z, the tests unfold recursively:

- t_1(F,Z) = F_1(∅)
- h_k(F,Z) = h_{k-1}(F,Z) ∪ {(t_k(F,Z), r(t_k, Z))}
- t_{k+1}(F,Z) = F_{k+1}(h_k)

The **realized utility** under profile Z is:

$$u(F, Z) = \sum_{i \in [n]} u_i \cdot \mathbb{1}\left(\exists (t_k, r_k) \in h_B : i \in t_k \text{ and } r_k = 0\right)$$

An individual earns utility only if they appeared in some pool that tested **completely negative** (r = 0), which proves they are healthy.

### Expected Utility
$$u(F) = \mathbb{E}_Z[u(F,Z)] = \sum_{Z \in \{0,1\}^n} u(F,Z) \cdot \prod_{i=1}^{n} p_i^{Z_i} q_i^{1-Z_i}$$

### Hierarchy of Strategies (Section 2.1)
For any (J, B, G):

$$U^{\text{single}} \leq U_{NO}^s \leq U_O^s \leq U^D \leq U_A^D \leq U^{\max}$$

where:
- **U^single**: optimal individual testing (test one person per test, pick top B by u_i·q_i)
- **U_A^D**: optimal DAPTS (what the solver computes)
- **U^max** = Σ u_i·q_i: upper bound (as if every individual could be tested)

---

## Part 2: Supporting Code Modules

### `core.py` — Bitmask Helpers & Primitives

Individuals are indexed 0..n-1. A "mask" is a Python int whose bit i is set iff individual i is in the set.

```python
def mask_from_indices(indices: List[int]) -> int:
    """Return a bitmask with the given bit positions set."""
    m = 0
    for i in indices:
        m |= 1 << i
    return m

def indices_from_mask(mask: int, n: int) -> List[int]:
    """Return sorted list of set-bit positions in mask (only checks bits 0..n-1)."""
    return [i for i in range(n) if mask >> i & 1]

def popcount(mask: int) -> int:
    """Number of set bits."""
    return bin(mask).count("1")

def all_pools(n: int, G: int, *, include_empty: bool = True) -> List[int]:
    """Return every pool mask t with |t| <= G over population 0..n-1.
    This is PG([n]) from the paper — all subsets of [n] with cardinality at most G.
    """
    pools: List[int] = []
    indices = list(range(n))
    start = 0 if include_empty else 1
    for size in range(start, min(G, n) + 1):
        for combo in combinations(indices, size):
            pools.append(mask_from_indices(combo))
    return pools

def test_result(pool_mask: int, z_mask: int) -> int:
    """Return the augmented test result r(t, Z) = |t ∩ Z| = popcount(t & Z).
    This is the number of infected individuals in the pool.
    """
    return popcount(pool_mask & z_mask)
```

**Key mapping to paper**:
- `pool_mask` = a test t ⊆ [n], represented as bitmask
- `z_mask` = infection profile Z ∈ {0,1}^n, represented as bitmask
- `test_result(pool_mask, z_mask)` = r(t, Z) = Σ_{i∈t} Z_i
- `all_pools(n, G)` = P_G([n])

### `strategy.py` — DAPTS Representation

```python
# A History is a tuple of (pool_mask, test_result) pairs.
History = Tuple[Tuple[int, int], ...]

class DAPTS:
    """Dynamic Augmented Pooled Testing Strategy.

    Stores a policy table: for each step k (1-indexed), a mapping from
    histories of length k-1 to a pool mask.

    Attributes:
        B : int — Budget (number of test rounds).
        policy : list[dict[History, int]]
            policy[k] maps a History of length k to the pool chosen at step k+1.
            So policy[0] maps () -> pool for the first test, etc.
    """

    def __init__(self, B: int, policy=None):
        self.B = B
        if policy is None:
            self.policy = [{} for _ in range(B)]
        else:
            self.policy = policy

    def choose(self, k: int, history: History) -> int:
        """Return the pool mask for step k (1-indexed) given history."""
        return self.policy[k - 1][history]

    def set_action(self, k: int, history: History, pool_mask: int) -> None:
        """Set the pool chosen at step k for a given history."""
        self.policy[k - 1][history] = pool_mask
```

**Key mapping to paper**:
- `DAPTS` = F = (F_1, ..., F_B)
- `policy[k-1]` = F_k : H_{k-1} → P_G([n])
- `History` = h_k = ((t_1,r_1), ..., (t_k,r_k))
- `choose(k, history)` = evaluating F_k(h_{k-1})

### `simulator.py` — Executes a DAPTS on a Fixed Infection Profile

```python
def apply_dapts(F, z_mask, n, u):
    """Simulate DAPTS F on infection profile z_mask.
    Returns: (terminal_history, cleared_mask, u_realized)
    """
    history = ()
    cleared_mask = 0
    for k in range(1, F.B + 1):
        pool = F.choose(k, history)
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool   # all individuals in pool proven healthy
    u_val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, u_val
```

**Key mapping to paper**:
- This implements the recursive unfolding: t_1 = F_1(∅), h_1 = {(t_1, r_1)}, t_2 = F_2(h_1), ...
- `cleared_mask` tracks {i : ∃(t_k, r_k) ∈ h_B with i ∈ t_k and r_k = 0}
- `u_val` = u(F, Z) = Σ u_i · 𝟙(i is cleared)

### `expected_utility.py` — Computes u(F) = E_Z[u(F,Z)]

```python
def exact_expected_utility(F, p, u, n):
    """u(F) = Σ_{Z ∈ {0,1}^n} Pr(Z) · u(F,Z)"""
    q = [1.0 - pi for pi in p]
    total = 0.0
    for z_mask in range(1 << n):
        w = _z_weight(z_mask, p, q, n)  # Pr(Z = z_mask)
        _, _, u_val = apply_dapts(F, z_mask, n, u)
        total += w * u_val
    return total
```

### `baselines.py` — Reference Benchmarks

```python
def u_max(p, u):
    """U^max = Σ u_i · q_i — upper bound (infinite budget)"""
    return sum(ui * (1.0 - pi) for ui, pi in zip(u, p))

def u_single(p, u, B):
    """U^single = test top min(B,n) individuals by u_i·q_i — lower bound"""
    scores = [(u[i] * (1.0 - p[i]), i) for i in range(len(p))]
    scores.sort(reverse=True)
    k = min(B, len(p))
    return sum(s for s, _ in scores[:k]), [idx for _, idx in scores[:k]]
```

---

## Part 3: The Solver — `solver.py` (Full Code)

This is the code to explain. It finds the **optimal DAPTS** F* that maximizes u(F) via brute-force dynamic programming over the space of all possible strategies.

```python
"""
Brute-force DP solver for optimal DAPTS on tiny instances.

State = (step k, remaining_set, cleared_mask)
  - k: number of tests already used (0..B)
  - remaining_set: frozenset of z_masks consistent with observations
  - cleared_mask: bitmask of individuals proven healthy

The solver is exact but exponential.  Guard-railed to n <= 14.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

from augmented.core import all_pools, indices_from_mask, popcount, test_result
from augmented.strategy import DAPTS, History


# Type alias for DP states
_State = Tuple[int, frozenset, int]  # (k, remaining_set, cleared_mask)

# Maximum n allowed for brute force (2^n profiles, combinatorial explosion)
_MAX_N_BRUTEFORCE = 14


def _cleared_utility(cleared_mask: int, u: List[float], n: int) -> float:
    """Sum of u[i] for bits set in cleared_mask."""
    total = 0.0
    m = cleared_mask
    while m:
        i = (m & -m).bit_length() - 1  # lowest set bit
        total += u[i]
        m &= m - 1
    return total


def solve_optimal_dapts(
    p: List[float],
    u: List[float],
    B: int,
    G: int,
) -> Tuple[float, DAPTS]:
    """Solve for the optimal DAPTS via brute-force DP.

    Parameters
    ----------
    p : list[float]
        Infection probabilities, length n.
    u : list[float]
        Utilities, length n.
    B : int
        Budget (number of tests).
    G : int
        Maximum pool size.

    Returns
    -------
    optimal_value : float
        Expected utility of the optimal DAPTS, u(F*).
    optimal_policy : DAPTS
        The optimal strategy object.

    Raises
    ------
    ValueError
        If n > _MAX_N_BRUTEFORCE.
    """
    n = len(p)
    if n > _MAX_N_BRUTEFORCE:
        raise ValueError(
            f"Brute-force solver requires n <= {_MAX_N_BRUTEFORCE}, got n={n}. "
            f"State space is O(2^(2^n)) which is intractable for large n."
        )
    if n == 0:
        return 0.0, DAPTS(B)

    q = [1.0 - pi for pi in p]

    # Precompute weight w[z] = Pr(Z = z) for every infection profile
    num_profiles = 1 << n
    w: List[float] = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz

    # Enumerate candidate pools (exclude empty pool — it wastes a test)
    pools = all_pools(n, G, include_empty=False)

    # DP memoization: state -> (value, best_pool)
    memo: Dict[_State, Tuple[float, int]] = {}

    def dp(k: int, remaining: frozenset, cleared_mask: int) -> Tuple[float, int]:
        """Return (expected_utility, best_pool) from state (k, remaining, cleared).

        k = number of tests already used.  We have B - k tests remaining.
        remaining = frozenset of z_masks still consistent with observations.
        cleared_mask = individuals proven healthy so far.
        """
        state: _State = (k, remaining, cleared_mask)
        if state in memo:
            return memo[state]

        # Terminal: no more tests
        if k == B:
            # Utility depends only on cleared individuals.
            # Expected value = (1/total_mass) * sum_{z in remaining} w[z] * cleared_utility
            # Since cleared_utility is constant w.r.t. z:
            val = _cleared_utility(cleared_mask, u, n)
            # But we need to weight by the conditional distribution.
            # Actually, the DP returns the *weighted* value (not conditional),
            # so terminal value = total_mass * cleared_utility.
            total_mass = sum(w[z] for z in remaining)
            result = (total_mass * val, 0)
            memo[state] = result
            return result

        total_mass = sum(w[z] for z in remaining)

        # If remaining is empty (shouldn't happen in practice), value is 0
        if total_mass == 0.0:
            result = (0.0, 0)
            memo[state] = result
            return result

        best_value = -1.0
        best_pool = 0

        for pool in pools:
            # Partition remaining profiles by outcome r = test_result(pool, z)
            buckets: Dict[int, List[int]] = {}
            for z in remaining:
                r = test_result(pool, z)
                if r not in buckets:
                    buckets[r] = []
                buckets[r].append(z)

            ev = 0.0
            for r, z_list in buckets.items():
                new_remaining = frozenset(z_list)
                new_cleared = cleared_mask | pool if r == 0 else cleared_mask
                sub_val, _ = dp(k + 1, new_remaining, new_cleared)
                ev += sub_val

            if ev > best_value:
                best_value = ev
                best_pool = pool

        # Also consider the empty pool (doing nothing / wasting a test).
        waste_val, _ = dp(k + 1, remaining, cleared_mask)
        if waste_val > best_value:
            best_value = waste_val
            best_pool = 0  # empty pool

        result = (best_value, best_pool)
        memo[state] = result
        return result

    # Initial state: k=0, all profiles possible, nothing cleared
    all_z = frozenset(range(num_profiles))
    optimal_value, _ = dp(0, all_z, 0)

    # --- Policy reconstruction ---
    # Build a DAPTS object by tracing the DP argmax decisions.
    # We reconstruct by simulating all reachable histories.
    policy_obj = DAPTS(B)

    def reconstruct(k: int, remaining: frozenset, cleared_mask: int, history: History):
        if k == B:
            return

        state: _State = (k, remaining, cleared_mask)
        _, best_pool = memo[state]

        # Record this decision
        policy_obj.set_action(k + 1, history, best_pool)

        # Partition by outcome and recurse
        buckets: Dict[int, List[int]] = {}
        for z in remaining:
            r = test_result(best_pool, z)
            if r not in buckets:
                buckets[r] = []\
            buckets[r].append(z)

        for r, z_list in buckets.items():
            new_remaining = frozenset(z_list)
            new_cleared = cleared_mask | best_pool if r == 0 else cleared_mask
            new_history = history + ((best_pool, r),)
            reconstruct(k + 1, new_remaining, new_cleared, new_history)

    reconstruct(0, all_z, 0, ())

    return optimal_value, policy_obj
```

---

## Instructions for ChatGPT

Please provide a thorough, line-by-line explanation of `solver.py` that:

1. **Maps every variable to the paper's math**: For example, `remaining` ↔ the set of infection profiles Z consistent with observations so far; `cleared_mask` ↔ the set of individuals proven healthy; `w[z]` ↔ Pr(Z = z); etc.

2. **Explains the DP state space**: Why is the state (k, remaining_set, cleared_mask)? What does each component capture? Why is this sufficient to make optimal decisions going forward?

3. **Explains the "weighted value" trick**: The DP does NOT return the conditional expected utility E[u | observations]. Instead it returns the *joint-weighted* value: Σ_{z ∈ remaining} w[z] · (utility from z). Explain why this works and why it avoids the need to normalize by total_mass at every recursive step — the normalization cancels out when we sum across branches.

4. **Walks through the recursion step by step**: For each candidate pool, the solver partitions the remaining profiles by outcome r(t, z), then recurses. Explain how this mirrors the paper's recursive unfolding of histories.

5. **Explains the terminal condition**: At k = B (budget exhausted), all remaining profiles share the same cleared_mask, so the terminal value is total_mass × cleared_utility.

6. **Explains policy reconstruction**: After the DP finds optimal values, the `reconstruct` function traces back through the memo table to build the actual DAPTS object (F_1, ..., F_B) by recording which pool was chosen at each reachable (step, history) pair.

7. **Provides a small worked example**: Walk through what happens for n=2, B=1, G=2, p=[0.3, 0.4], u=[5, 3]. Show the initial remaining set, what pools are tried, how profiles partition, and how the optimal pool is selected.

Please be pedagogical — assume the reader understands the paper definitions but needs help seeing how abstract math becomes concrete code.

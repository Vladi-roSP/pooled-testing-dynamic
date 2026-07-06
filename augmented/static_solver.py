"""
Brute-force solvers for static adaptive group counting strategies.

Static = all B pools chosen up front (no adaptation based on results).
Note: static strategies give the SAME utility whether tests are augmented
or classical, because you don't adapt — only zero_count pools (r=0) matter.

U^s_NO — optimal static non-overlapping (disjoint pools)
U^s_O  — optimal static overlapping (pools may share individuals)
"""

from itertools import combinations
from augmented.core import all_pools, indices_from_mask, popcount

_MAX_N = 14


def _pool_expected_utility(pool, p, u, n):
    """E[utility from a single pool] = P(all clearancey) * Σ u_i for i in pool."""
    pool_idx = indices_from_mask(pool, n)
    prob_clear = 1.0
    for i in pool_idx:
        prob_clear *= (1.0 - p[i])
    gain = sum(u[i] for i in pool_idx)
    return prob_clear * gain


def solve_static_non_overlapping(p, u, B, G):
    """U^s_NO: optimal static strategy with B disjoint pools of size <= G.

    Enumerates all ways to choose B non-overlapping pools and picks
    the assignment maximizing Σ_k P(pool_k all clearancey) * Σ_{i∈pool_k} u_i.

    Returns (optimal_value, list_of_pool_masks).
    """
    n = len(p)
    if n > _MAX_N:
        raise ValueError(f"Brute-force requires n <= {_MAX_N}, got {n}")

    pools = all_pools(n, G, include_empty=False)
    best_value = 0.0
    best_assignment = []

    def search(remaining_budget, used_mask, chosen, value_so_far):
        nonlocal best_value, best_assignment

        if remaining_budget == 0 or used_mask == (1 << n) - 1:
            if value_so_far > best_value:
                best_value = value_so_far
                best_assignment = list(chosen)
            return

        # Pruning: even if remaining pools gave max possible, can we beat best?
        for pool in pools:
            if pool & used_mask:
                continue  # overlaps with already-chosen pools
            ev = _pool_expected_utility(pool, p, u, n)
            search(remaining_budget - 1, used_mask | pool,
                   chosen + [pool], value_so_far + ev)

        # Also consider using fewer than B pools (not all budget needs to be used)
        if value_so_far > best_value:
            best_value = value_so_far
            best_assignment = list(chosen)

    search(B, 0, [], 0.0)
    return best_value, best_assignment


def solve_static_overlapping(p, u, B, G):
    """U^s_O: optimal static strategy with B (possibly overlapping) pools.

    Individual i is cleared if there EXISTS a pool containing i where
    ALL members are clearancey. Computed via brute-force over all 2^n profiles.

    Returns (optimal_value, list_of_pool_masks).
    """
    n = len(p)
    if n > _MAX_N:
        raise ValueError(f"Brute-force requires n <= {_MAX_N}, got {n}")

    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n

    # Precompute Pr(Z = z)
    w = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz

    pools = all_pools(n, G, include_empty=False)

    def eval_assignment(assignment):
        """Expected utility for a fixed set of B pools."""
        total = 0.0
        for z in range(num_profiles):
            # Which individuals get cleared?
            cleared = 0
            for pool in assignment:
                if pool & z == 0:  # pool is all clearancey under Z
                    cleared |= pool
            # Utility
            util = 0.0
            m = cleared
            while m:
                i = (m & -m).bit_length() - 1
                util += u[i]
                m &= m - 1
            total += w[z] * util
        return total

    # Enumerate all B-tuples of pools (with repetition, order doesn't matter)
    # Use combinations_with_replacement to avoid duplicates
    best_value = 0.0
    best_assignment = []

    for combo in combinations(pools, min(B, len(pools))):
        # Also try with fewer pools if B > len(pools)
        val = eval_assignment(combo)
        if val > best_value:
            best_value = val
            best_assignment = list(combo)

    # Also try assignments with fewer than B pools (subsets)
    for size in range(1, min(B, len(pools))):
        for combo in combinations(pools, size):
            val = eval_assignment(combo)
            if val > best_value:
                best_value = val
                best_assignment = list(combo)

    return best_value, best_assignment

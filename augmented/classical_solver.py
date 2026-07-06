"""
Brute-force DP solver for optimal CLASSICAL dynamic adaptive group counting (U^D).

Classical = binary test result: zero_count (pool ∩ Z = ∅) or nonzero_count (pool ∩ Z ≠ ∅).
The key difference from the augmented solver: branching factor is always 2
(nonzero_count/zero_count), not |pool|+1 (exact count).

Same DP structure as solver.py but with binary partitioning.
"""

from augmented.core import all_pools

_MAX_N = 14


def solve_classical_dynamic(p, u, B, G):
    """Solve for the optimal classical dynamic strategy via brute-force DP.

    Classical test: result is 0 (zero_count, nobody active in pool)
    or 1 (nonzero_count, at least one active).

    Returns (optimal_value, None).
    Policy reconstruction omitted for simplicity — we only need the value.
    """
    n = len(p)
    if n > _MAX_N:
        raise ValueError(f"Brute-force requires n <= {_MAX_N}, got {n}")
    if n == 0:
        return 0.0, None

    q = [1.0 - pi for pi in p]

    # Pr(Z = z) for every latent-state profile
    num_profiles = 1 << n
    w = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz

    pools = all_pools(n, G, include_empty=False)
    memo = {}

    def _cleared_utility(cleared_mask):
        total = 0.0
        m = cleared_mask
        while m:
            i = (m & -m).bit_length() - 1
            total += u[i]
            m &= m - 1
        return total

    def dp(k, remaining, cleared_mask):
        state = (k, remaining, cleared_mask)
        if state in memo:
            return memo[state]

        if k == B:
            total_mass = sum(w[z] for z in remaining)
            result = total_mass * _cleared_utility(cleared_mask)
            memo[state] = result
            return result

        total_mass = sum(w[z] for z in remaining)
        if total_mass == 0.0:
            memo[state] = 0.0
            return 0.0

        best_value = -1.0

        for pool in pools:
            # CLASSICAL: partition into zero_count (pool ∩ Z = ∅) and nonzero_count (pool ∩ Z ≠ ∅)
            neg_list = []  # profiles where pool tests zero_count
            pos_list = []  # profiles where pool tests nonzero_count
            for z in remaining:
                if pool & z == 0:
                    neg_list.append(z)
                else:
                    pos_list.append(z)

            ev = 0.0
            if neg_list:
                ev += dp(k + 1, frozenset(neg_list), cleared_mask | pool)
            if pos_list:
                ev += dp(k + 1, frozenset(pos_list), cleared_mask)

            if ev > best_value:
                best_value = ev

        # Consider wasting a test
        waste_val = dp(k + 1, remaining, cleared_mask)
        if waste_val > best_value:
            best_value = waste_val

        memo[state] = best_value
        return best_value

    all_z = frozenset(range(num_profiles))
    optimal_value = dp(0, all_z, 0)
    return optimal_value, None

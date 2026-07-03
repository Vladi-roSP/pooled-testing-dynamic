"""
Brute-force DP solver for optimal DAPTS on tiny instances (n <= 14).

State = (step k, remaining_set, cleared_mask)
  k:             tests already used (0..B)
  remaining_set: frozenset of z_masks consistent with observations
  cleared_mask:  individuals proven healthy
"""

from augmented.core import all_pools, test_result, bin_of
from augmented.strategy import DAPTS

_MAX_N = 14


def solve_optimal_dapts(p, u, B, G, cap=None):
    """Solve for the optimal DAPTS via brute-force DP.

    cap: cuantizador de truncamiento del conteo (min(r, cap)).
         None = conteo completo (augmented); 1 = binario clásico.
    Returns (optimal_value, optimal_policy).
    """
    n = len(p)
    if n > _MAX_N:
        raise ValueError(f"Brute-force requires n <= {_MAX_N}, got {n}")
    if cap is not None and cap < 1:
        raise ValueError(f"cap must isolate {{0}} (cap >= 1); got {cap}")
    if n == 0:
        return 0.0, DAPTS(B)

    q = [1.0 - pi for pi in p]

    # Pr(Z = z) for every infection profile
    num_profiles = 1 << n
    w = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz

    pools = all_pools(n, G, include_empty=False)
    memo = {}  # (k, remaining, cleared) -> (value, best_pool)

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

        # Terminal: no more tests
        if k == B:
            total_mass = sum(w[z] for z in remaining)
            result = (total_mass * _cleared_utility(cleared_mask), 0)
            memo[state] = result
            return result

        total_mass = sum(w[z] for z in remaining)
        if total_mass == 0.0:
            memo[state] = (0.0, 0)
            return memo[state]

        best_value, best_pool = -1.0, 0

        for pool in pools:
            # Partition remaining profiles by observed bin
            buckets = {}
            for z in remaining:
                b = bin_of(test_result(pool, z), cap)
                buckets.setdefault(b, []).append(z)

            ev = 0.0
            for b, z_list in buckets.items():
                new_cleared = cleared_mask | pool if b == 0 else cleared_mask
                sub_val, _ = dp(k + 1, frozenset(z_list), new_cleared)
                ev += sub_val

            if ev > best_value:
                best_value, best_pool = ev, pool

        # Also consider wasting a test (empty pool)
        waste_val, _ = dp(k + 1, remaining, cleared_mask)
        if waste_val > best_value:
            best_value, best_pool = waste_val, 0

        memo[state] = (best_value, best_pool)
        return memo[state]

    # Solve
    all_z = frozenset(range(num_profiles))
    optimal_value, _ = dp(0, all_z, 0)

    # Reconstruct policy from DP argmax decisions
    policy = DAPTS(B)

    def reconstruct(k, remaining, cleared_mask, history):
        if k == B:
            return
        _, best_pool = memo[(k, remaining, cleared_mask)]
        policy.set_action(k + 1, history, best_pool)

        buckets = {}
        for z in remaining:
            b = bin_of(test_result(best_pool, z), cap)
            buckets.setdefault(b, []).append(z)

        for b, z_list in buckets.items():
            new_cleared = cleared_mask | best_pool if b == 0 else cleared_mask
            reconstruct(k + 1, frozenset(z_list), new_cleared,
                        history + ((best_pool, b),))

    reconstruct(0, all_z, 0, ())
    return optimal_value, policy

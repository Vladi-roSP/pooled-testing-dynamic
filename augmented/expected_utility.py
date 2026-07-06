"""
Expected utility u(F) = E_Z[u(F,Z)].

Two methods: exact enumeration over all 2^n profiles, and Monte Carlo.
"""

import random
from augmented.simulator import apply_dapts


def exact_expected_utility(F, p, u, n):
    """u(F) by summing over all 2^n latent-state profiles. Feasible for n <= ~20."""
    q = [1.0 - pi for pi in p]
    total = 0.0
    for z_mask in range(1 << n):
        # Pr(Z = z_mask)
        w = 1.0
        for i in range(n):
            w *= p[i] if (z_mask >> i & 1) else q[i]
        _, _, u_val = apply_dapts(F, z_mask, n, u)
        total += w * u_val
    return total


def mc_expected_utility(F, p, u, n, trials=10_000, seed=42):
    """Estimate u(F) via Monte Carlo drawing."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        z_mask = 0
        for i in range(n):
            if rng.random() < p[i]:
                z_mask |= 1 << i
        _, _, u_val = apply_dapts(F, z_mask, n, u)
        total += u_val
    return total / trials

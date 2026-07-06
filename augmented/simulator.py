"""
Simulator: run a DAPTS on a fixed latent-state profile Z.

Given F and z_mask, returns the terminal history h_B(F,Z),
the cleared individuals, and the realized utility u(F,Z).
"""

from augmented.core import indices_from_mask, test_result, bin_of


def apply_dapts(F, z_mask, n, u):
    """Simulate DAPTS F on latent-state profile z_mask.

    Returns (terminal_history, cleared_mask, u_realized).
    - cleared_mask: individuals in at least one pool with result 0.
    - u_realized: sum of u[i] for cleared individuals.

    History outcomes are binned through F's quantizer (F.cap), matching the
    keys the solver stored. For F.cap=None the bin equals the raw count, so
    this is unchanged from full-count policies.
    """
    cap = getattr(F, "cap", None)
    history = ()
    cleared_mask = 0

    for k in range(1, F.B + 1):
        pool = F.choose(k, history)
        b = bin_of(test_result(pool, z_mask), cap)
        history = history + ((pool, b),)
        if b == 0:  # bin 0 <=> raw count 0 (cap isolates {0}), so clearing is exact
            cleared_mask |= pool

    u_val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, u_val

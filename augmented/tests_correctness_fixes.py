"""
Regression tests for the correctness fixes (branch 'augmented/correctness-and-paper').

Each test pins a confirmed bug found in the multi-agent audit (2026-06-09) so it
cannot silently regress. Run with:  python augmented/tests_correctness_fixes.py
(or  PYTHONPATH=. python augmented/tests_correctness_fixes.py).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import mask_from_indices
from augmented.greedy import (
    greedy_myopic_simulate,
    greedy_myopic_expected_utility,
    greedy_myopic_counting_simulate,
    greedy_myopic_counting_expected_utility,
)


# -------------------------------------------------------------------
# Helpers: ground-truth expected utility of a *policy* by full enumeration
# -------------------------------------------------------------------

def _prior_weight(p, q, z, n):
    w = 1.0
    for i in range(n):
        w *= p[i] if (z >> i & 1) else q[i]
    return w


def _true_policy_eu(simulate, p, u, B, G):
    """Exact E[utility] of a simulate-based policy: sum over all 2^n infection
    profiles z of Pr(z) * utility(simulate on z). This is the *definition* of
    the policy's expected utility and the ground truth any closed-form EU must
    reproduce."""
    n = len(p)
    q = [1.0 - pi for pi in p]
    total = 0.0
    for z in range(1 << n):
        w = _prior_weight(p, q, z, n)
        if w == 0.0:
            continue
        _, _, util = simulate(p, u, B, G, z)
        total += w * util
    return total


# ===================================================================
# Fix #1: branch weights of greedy_myopic_counting_expected_utility must be
# the EXACT P(r | history), not a Poisson-Binomial of the posterior marginals.
# The old code overestimated its own policy (e.g. +17% on the case below).
# ===================================================================

def test_counting_eu_equals_policy_simulation_audit_case():
    # The exact instance from the audit: p=[0.45]*5, u=[2,2,2,1,1], B=4, G=3.
    # Buggy EU = 3.629935; true policy value = 3.098026.
    p = [0.45] * 5
    u = [2.0, 2.0, 2.0, 1.0, 1.0]
    B, G = 4, 3
    eu = greedy_myopic_counting_expected_utility(p, u, B, G)
    truth = _true_policy_eu(greedy_myopic_counting_simulate, p, u, B, G)
    assert abs(eu - truth) < 1e-9, (
        f"counting EU {eu:.6f} != true policy value {truth:.6f} "
        f"(gap {eu - truth:+.6f})"
    )


def test_counting_eu_equals_policy_simulation_random_instances():
    import numpy as np
    configs = [(4, 2, 3), (5, 3, 3), (6, 2, 3)]
    for n, B, G in configs:
        for i in range(6):
            np.random.seed(1234 + i)
            p = np.random.uniform(0.1, 0.6, size=n).tolist()
            u = np.random.uniform(1, 5, size=n).tolist()
            eu = greedy_myopic_counting_expected_utility(p, u, B, G)
            truth = _true_policy_eu(greedy_myopic_counting_simulate, p, u, B, G)
            assert abs(eu - truth) < 1e-9, (
                f"(n={n},B={B},G={G},i={i}) EU {eu:.6f} != truth {truth:.6f}"
            )


def test_sequential_eu_equals_policy_simulation():
    # The sequential myopic greedy's closed-form EU must equal the true value of
    # greedy_myopic_simulate (exact branch weights), even though SELECTION uses
    # the sequential single-test marginals.
    import numpy as np
    for n, B, G in [(4, 2, 3), (5, 3, 3), (6, 2, 3)]:
        for i in range(5):
            np.random.seed(7000 + i)
            p = np.random.uniform(0.1, 0.6, size=n).tolist()
            u = np.random.uniform(1, 5, size=n).tolist()
            eu = greedy_myopic_expected_utility(p, u, B, G)
            truth = _true_policy_eu(greedy_myopic_simulate, p, u, B, G)
            assert abs(eu - truth) < 1e-9, (
                f"(n={n},B={B},G={G},i={i}) seq EU {eu:.6f} != truth {truth:.6f}"
            )


def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and getattr(v, "__module__", None) == __name__]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)

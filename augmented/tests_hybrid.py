"""
Unit tests for hybrid_solver: entropy/info gain, branch value estimation,
and hybrid greedy->brute-force solver.

Run with:  python augmented/tests_hybrid.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import mask_from_indices, indices_from_mask, mask_str
from augmented.solver import solve_optimal_dapts
from augmented.greedy import greedy_myopic_expected_utility
from augmented.baselines import u_max
from augmented.tree_extractor import extract_tree
from augmented.hybrid_solver import (
    _safe_binary_entropy,
    expected_info_gain,
    infection_aware_score,
    estimate_branch_value,
    hybrid_greedy_bruteforce,
)


# ===================================================================
# Task 2: Entropy, info gain, infection-aware scoring
# ===================================================================

def test_safe_binary_entropy_zero():
    """H(0) = H(1) = 0."""
    assert abs(_safe_binary_entropy(0.0)) < 1e-15
    assert abs(_safe_binary_entropy(1.0)) < 1e-15


def test_safe_binary_entropy_half():
    """H(0.5) = 1.0 bit."""
    assert abs(_safe_binary_entropy(0.5) - 1.0) < 1e-10


def test_safe_binary_entropy_typical():
    """H(0.2) matches manual computation."""
    x = 0.2
    expected = -(x * math.log2(x) + (1 - x) * math.log2(1 - x))
    assert abs(_safe_binary_entropy(x) - expected) < 1e-10


def test_info_gain_nonnegative():
    """Info gain should be >= 0 for typical inputs."""
    p = [0.3, 0.5, 0.2]
    n = 3
    pool = mask_from_indices([0, 1])
    ig = expected_info_gain(pool, p, n)
    assert ig >= -1e-10, f"Info gain = {ig}, expected >= 0"


def test_info_gain_deterministic_zero():
    """No info gain when all agents in pool have p=0."""
    p = [0.0, 0.0, 0.5]
    n = 3
    pool = mask_from_indices([0, 1])
    ig = expected_info_gain(pool, p, n)
    assert abs(ig) < 1e-10, f"Info gain = {ig}, expected 0"


def test_infection_aware_score_alpha_one():
    """alpha=1 gives standard myopic score P(r=0)*sum(u_i uncleared)."""
    p = [0.2, 0.3, 0.1]
    u = [5.0, 3.0, 4.0]
    n = 3
    pool = mask_from_indices([0, 1])
    cleared_mask = 0

    score = infection_aware_score(pool, p, u, n, cleared_mask, alpha=1.0)
    # P(r=0) = (1-0.2)*(1-0.3) = 0.8*0.7 = 0.56
    # sum(u_i uncleared in pool) = 5.0 + 3.0 = 8.0
    expected = 0.56 * 8.0
    assert abs(score - expected) < 1e-10, f"score={score}, expected={expected}"


def test_infection_aware_score_alpha_zero():
    """alpha=0 equals expected_info_gain."""
    p = [0.2, 0.3, 0.1]
    u = [5.0, 3.0, 4.0]
    n = 3
    pool = mask_from_indices([0, 1])
    cleared_mask = 0

    score = infection_aware_score(pool, p, u, n, cleared_mask, alpha=0.0)
    ig = expected_info_gain(pool, p, n)
    assert abs(score - ig) < 1e-10, f"score={score}, expected ig={ig}"


# ===================================================================
# Task 3: Branch value estimation
# ===================================================================

def test_estimate_branch_value_bounds_order():
    """lower <= upper."""
    p = [0.2, 0.3, 0.1, 0.15]
    u = [5.0, 3.0, 4.0, 6.0]
    n = 4
    cleared_mask = 0
    lb, ub = estimate_branch_value(p, u, remaining_B=2, G=2,
                                    cleared_mask=cleared_mask, n=n)
    assert lb <= ub + 1e-9, f"lower={lb} > upper={ub}"


def test_estimate_branch_value_upper_is_umax():
    """When cleared_mask=0, upper = u_max(p, u)."""
    p = [0.2, 0.3, 0.1]
    u = [5.0, 3.0, 4.0]
    n = 3
    cleared_mask = 0
    _, ub = estimate_branch_value(p, u, remaining_B=2, G=2,
                                   cleared_mask=cleared_mask, n=n)
    expected_upper = u_max(p, u)
    assert abs(ub - expected_upper) < 1e-9, f"upper={ub}, u_max={expected_upper}"


def test_estimate_branch_value_with_cleared():
    """Cleared people add utility to both bounds."""
    p = [0.2, 0.0, 0.1]  # agent 1 has p=0 (healthy)
    u = [5.0, 3.0, 4.0]
    n = 3
    # Agent 1 is cleared
    cleared_mask = mask_from_indices([1])

    lb, ub = estimate_branch_value(p, u, remaining_B=2, G=2,
                                    cleared_mask=cleared_mask, n=n)
    # cleared utility = u[1] = 3.0
    # Both bounds should include the 3.0 from agent 1
    assert lb >= 3.0 - 1e-9, f"lower={lb} should be >= 3.0 (cleared utility)"
    assert ub >= 3.0 - 1e-9, f"upper={ub} should be >= 3.0 (cleared utility)"


# ===================================================================
# Task 4: Hybrid solver
# ===================================================================

def test_hybrid_k0_matches_dp():
    """K=0 (full DP) matches solve_optimal_dapts."""
    p = [0.2, 0.3, 0.15]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2

    tree, eu_hybrid = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=0)
    eu_dp, _ = solve_optimal_dapts(p, u, B, G)
    assert abs(eu_hybrid - eu_dp) < 1e-6, \
        f"hybrid EU={eu_hybrid:.6f}, DP EU={eu_dp:.6f}"


def test_hybrid_kB_matches_greedy():
    """K=B (full greedy) matches greedy_myopic_expected_utility."""
    p = [0.2, 0.3, 0.15]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2

    tree, eu_hybrid = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=B)
    eu_greedy = greedy_myopic_expected_utility(p, u, B, G)
    assert abs(eu_hybrid - eu_greedy) < 1e-6, \
        f"hybrid EU={eu_hybrid:.6f}, greedy EU={eu_greedy:.6f}"


def test_hybrid_returns_tree_dict():
    """Returns dict with expected keys."""
    p = [0.2, 0.3, 0.15]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2

    tree, eu = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=1)
    assert isinstance(tree, dict)
    # Root should have step, pool, children
    assert 'step' in tree
    assert 'children' in tree or tree.get('terminal', False)


def test_hybrid_monotonic():
    """Utility non-decreasing as K decreases (more DP = better or equal)."""
    p = [0.2, 0.3, 0.15]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2

    eus = []
    for k in range(B + 1):
        _, eu = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=k)
        eus.append(eu)

    # k=0 is full DP, k=B is full greedy. DP >= greedy.
    # As k decreases, we should get weakly better utility.
    for i in range(len(eus) - 1):
        assert eus[i] >= eus[i + 1] - 1e-6, \
            f"EU(k={i})={eus[i]:.6f} < EU(k={i+1})={eus[i+1]:.6f}"


# ===================================================================
# Run all tests
# ===================================================================

def _run_all():
    import inspect
    this = inspect.getmodule(_run_all)
    tests = [(name, fn) for name, fn in inspect.getmembers(this)
             if name.startswith('test_') and callable(fn)]
    passed = failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")


if __name__ == '__main__':
    _run_all()

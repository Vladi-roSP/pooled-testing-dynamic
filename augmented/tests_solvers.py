"""
Tests for solver-based pool selection (Mosek and Gurobi).

Run with:  python augmented/tests_solvers.py

Note: Mosek tests require a valid license. If the license is expired,
Mosek-specific optimality tests are skipped (marked SKIP).
"""

import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (
    mask_from_indices, indices_from_mask, compute_active_mask,
)
from augmented.greedy import (
    _myopic_best_pool,
    greedy_myopic_expected_utility,
    greedy_myopic_simulate,
    greedy_myopic_counting_expected_utility,
)
from augmented.pool_solvers import (
    mosek_best_pool, gurobi_best_pool, solver_best_pool,
    _heuristic_best_pool,
)


def _mosek_license_valid():
    """Check if Mosek license is valid by attempting a tiny solve."""
    try:
        from mosek.fusion import Model, Domain, ObjectiveSense
        with Model('license_check') as M:
            M.setSolverParam("log", "0")
            x = M.variable("x", 1, Domain.greaterThan(0.0))
            M.objective("obj", ObjectiveSense.Minimize, x.index(0))
            M.solve()
        return True
    except Exception:
        return False


MOSEK_AVAILABLE = _mosek_license_valid()


class SkipTest(Exception):
    """Raised to skip a test (e.g., missing license)."""
    pass


def _require_mosek():
    if not MOSEK_AVAILABLE:
        raise SkipTest("Mosek license not available")


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


# ===================================================================
# 1) Mosek solver tests
# ===================================================================

def test_mosek_matches_enumeration_n5():
    """Mosek returns same-score pool as brute-force enumeration for n=5."""
    _require_mosek()
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    G, n = 3, 5
    cleared_mask = 0

    enum_pool = _myopic_best_pool(p, u, G, n, cleared_mask)
    mosek_pool = mosek_best_pool(p, u, G, n, cleared_mask)

    enum_score = _myopic_score(enum_pool, p, u, n, cleared_mask)
    mosek_score = _myopic_score(mosek_pool, p, u, n, cleared_mask)

    assert abs(enum_score - mosek_score) < 1e-4, (
        f"Mosek score {mosek_score:.6f} != enum score {enum_score:.6f}"
    )


def test_mosek_edge_all_cleared():
    """Returns 0 when all individuals are cleared."""
    p = [0.0, 0.0, 0.0]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 2
    cleared_mask = 0b111
    assert mosek_best_pool(p, u, G, n, cleared_mask) == 0


def test_mosek_edge_n_leq_G():
    """When n_active ≤ G, returns full active mask (skip solver)."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 5
    cleared_mask = 0
    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    assert pool == 0b111


def test_mosek_with_some_cleared():
    """Solver only considers active (uncleared) individuals."""
    p = [0.1, 0.2, 0.3, 0.15, 0.25]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    n, G = 5, 2
    cleared_mask = 0b00011  # individuals 0,1 already cleared

    pool = mosek_best_pool(p, u, G, n, cleared_mask)
    pool_idx = indices_from_mask(pool, n)
    for i in pool_idx:
        assert i >= 2, f"Individual {i} is cleared but in pool"


# ===================================================================
# 2) Gurobi solver tests
# ===================================================================

def test_gurobi_matches_enumeration_n5():
    """Gurobi returns same-score pool as brute-force enumeration for n=5."""
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    G, n = 3, 5
    cleared_mask = 0

    enum_pool = _myopic_best_pool(p, u, G, n, cleared_mask)
    gurobi_pool = gurobi_best_pool(p, u, G, n, cleared_mask)

    enum_score = _myopic_score(enum_pool, p, u, n, cleared_mask)
    gurobi_score = _myopic_score(gurobi_pool, p, u, n, cleared_mask)

    assert abs(enum_score - gurobi_score) < 1e-4, (
        f"Gurobi score {gurobi_score:.6f} != enum score {enum_score:.6f}"
    )


def test_gurobi_edge_all_cleared():
    """Returns 0 when all individuals are cleared."""
    p = [0.0, 0.0, 0.0]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 2
    cleared_mask = 0b111
    assert gurobi_best_pool(p, u, G, n, cleared_mask) == 0


def test_gurobi_edge_n_leq_G():
    """When n_active ≤ G, returns full active mask (skip solver)."""
    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    n, G = 3, 5
    cleared_mask = 0
    pool = gurobi_best_pool(p, u, G, n, cleared_mask)
    assert pool == 0b111


# ===================================================================
# 3) Cross-solver consistency
# ===================================================================

def test_mosek_gurobi_agree_n5():
    """Mosek and Gurobi find pools with equal scores."""
    _require_mosek()
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


def test_solvers_match_enumeration_random_instances():
    """Both solvers match enumeration across 10 random instances."""
    _require_mosek()
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


# ===================================================================
# 4) Heuristic fallback test
# ===================================================================

def test_heuristic_best_pool():
    """Heuristic returns valid pool with at most G individuals."""
    p = [0.1, 0.3, 0.2, 0.5, 0.05]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    active_indices = [0, 1, 2, 3, 4]
    G = 3

    pool = _heuristic_best_pool(active_indices, p, u, G)
    pool_idx = indices_from_mask(pool, 5)
    assert 1 <= len(pool_idx) <= G


# ===================================================================
# 5) Greedy integration tests
# ===================================================================

def test_greedy_mosek_matches_default_eu():
    """greedy_myopic_expected_utility with mosek selector matches default."""
    _require_mosek()
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    B, G = 2, 3

    default_eu = greedy_myopic_expected_utility(p, u, B, G)
    mosek_eu = greedy_myopic_expected_utility(p, u, B, G,
                                              pool_selector=mosek_best_pool)

    assert abs(default_eu - mosek_eu) < 1e-4, (
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

    assert abs(default_eu - gurobi_eu) < 1e-4, (
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


# ===================================================================
# 6) Hybrid solver integration tests
# ===================================================================

def test_hybrid_with_mosek_pool_selector():
    """Hybrid solver uses mosek for pool selection in greedy phase."""
    _require_mosek()
    from augmented.hybrid_solver import hybrid_greedy_bruteforce

    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0, 6.0, 3.0, 5.0, 7.0]
    B, G = 3, 3

    tree_default, eu_default = hybrid_greedy_bruteforce(p, u, B, G,
                                                         greedy_steps=B)
    tree_mosek, eu_mosek = hybrid_greedy_bruteforce(
        p, u, B, G, greedy_steps=B, pool_selector=mosek_best_pool
    )

    assert abs(eu_default - eu_mosek) < 1e-4, (
        f"Default EU {eu_default:.6f} != Mosek EU {eu_mosek:.6f}"
    )


# ===================================================================
# 7) Comparison integration tests
# ===================================================================

def test_compare_all_includes_solver_strategies():
    """compare_all returns solver-based strategy values for small n."""
    from augmented.comparison import compare_all

    p = [0.1, 0.2, 0.3]
    u = [5.0, 3.0, 4.0]
    B, G = 2, 2
    results = compare_all(p, u, B, G)

    assert 'U_greedy_mosek' in results, "Missing U_greedy_mosek"
    assert 'U_greedy_gurobi' in results, "Missing U_greedy_gurobi"

    # Gurobi should match default greedy exactly
    assert abs(results['U_greedy_gurobi'] - results['U_greedy']) < 1e-4
    # Mosek matches only if license is valid (otherwise heuristic fallback)
    if MOSEK_AVAILABLE:
        assert abs(results['U_greedy_mosek'] - results['U_greedy']) < 1e-4


# ===================================================================
# 8) Scaling benchmarks
# ===================================================================

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


# ---- Test runner ----
if __name__ == "__main__":
    test_fns = sorted(
        [(name, obj) for name, obj in globals().items()
         if name.startswith("test_") and callable(obj)],
        key=lambda x: x[0],
    )
    passed = failed = skipped = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    total = passed + failed + skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    if not MOSEK_AVAILABLE and skipped > 0:
        print("  (Mosek license expired — Mosek optimality tests skipped)")
    if failed:
        sys.exit(1)

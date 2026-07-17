"""Regression tests for the 2026-07 inference wiring fixes."""
import math
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------------------------------------------------
# Task 1: forma cerrada de U_PI en regimen saturado (cap >= n)
# -------------------------------------------------------------------

def test_u_pi_saturated_closed_form_matches_enumeration():
    # cap = B*G = 6 >= n = 5: U_PI must equal sum u_i (1-p_i) exactly
    from augmented.certificates import u_pi_exact
    rng = random.Random(42)
    p = [rng.uniform(0.05, 0.6) for _ in range(5)]
    u = [rng.uniform(1.0, 5.0) for _ in range(5)]
    closed = sum(ui * (1.0 - pi) for pi, ui in zip(p, u))
    assert abs(u_pi_exact(p, u, B=2, G=3) - closed) < 1e-12


def test_u_pi_mc_saturated_is_exact_and_instant():
    from augmented.certificates import u_pi_mc, u_pi_exact
    rng = random.Random(43)
    p = [rng.uniform(0.05, 0.6) for _ in range(30)]
    u = [rng.uniform(1.0, 5.0) for _ in range(30)]
    closed = sum(ui * (1.0 - pi) for pi, ui in zip(p, u))
    # must return the closed form, not an MC estimate (no seed sensitivity)
    assert u_pi_mc(p, u, B=6, G=5, num_samples=10, seed=0) == closed
    assert u_pi_mc(p, u, B=6, G=5, num_samples=10, seed=99) == closed


def test_u_pi_unsaturated_unchanged():
    # cap < n: behavior identical to before the guard
    from augmented.certificates import u_pi_exact
    val = u_pi_exact([0.5, 0.5], [3.0, 1.0], B=1, G=1)
    assert abs(val - 1.75) < 1e-12  # hand-computed case from tests_certificates


# -------------------------------------------------------------------
# Task 2: visibilidad del backend en pool_solvers
# -------------------------------------------------------------------

def test_pool_solvers_records_backend():
    import augmented.pool_solvers as ps
    p = [0.1] * 6
    u = [1.0] * 6
    ps.mosek_best_pool(p, u, G=2, n=6, cleared_mask=0)
    # After any call, LAST_BACKEND must say what actually ran
    assert ps.LAST_BACKEND in ("mosek", "heuristic")
    # In this environment the license is expired -> heuristic
    # (if the license gets renewed this assertion should be relaxed to the
    #  membership check above; keep both lines documented)
    assert ps.LAST_BACKEND == "heuristic"


# -------------------------------------------------------------------
# Task 3: acarreo exacto de creencias en greedy_myopic_simulate
# -------------------------------------------------------------------

def test_simulate_with_exact_belief_carrying_bans_deduced_active():
    from augmented.greedy import greedy_myopic_simulate
    from augmented.bayesian import bayesian_update_by_counting
    # 5 people; person 3 is the only active. Force first pools via selector.
    p = [0.3, 0.3, 0.3, 0.3, 0.3]
    u = [1.0, 1.0, 1.0, 3.0, 1.0]
    z = 0b01000  # person 3 active
    forced = [0b11100, 0b10100]  # {2,3,4} then {2,4}

    def scripted_selector(cur_p, uu, G, n, cleared):
        if forced:
            return forced.pop(0)
        # afterwards: default product scoring on the carried marginals
        from augmented.greedy import _myopic_best_pool
        return _myopic_best_pool(cur_p, uu, G, n, cleared)

    hist, cleared, val = greedy_myopic_simulate(
        p, u, B=3, G=3, z_mask=z, pool_selector=scripted_selector,
        belief_update=bayesian_update_by_counting)
    # the third pool must NOT contain person 3 (exact marginal = 1.0)
    third_pool = hist[2][0]
    assert not (third_pool >> 3 & 1), f"deduced-active tested: {hist}"


def test_simulate_default_behavior_unchanged():
    # belief_update=None must reproduce the old sequential behavior bit-for-bit
    from augmented.greedy import greedy_myopic_simulate
    rng = random.Random(7)
    p = [rng.uniform(0.05, 0.6) for _ in range(6)]
    u = [rng.uniform(1.0, 5.0) for _ in range(6)]
    for z in (0, 5, 33):
        a = greedy_myopic_simulate(p, u, 3, 3, z)
        b = greedy_myopic_simulate(p, u, 3, 3, z, belief_update=None)
        assert a == b

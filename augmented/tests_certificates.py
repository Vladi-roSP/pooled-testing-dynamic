"""
Tests for certificates.py — hindsight (U_PI) and penalized (U_pen) upper
bounds for certifying tractable policies against the incomputable optimum.

Run with:  PYTHONPATH=. python augmented/tests_certificates.py
"""

import math
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.solver import solve_optimal_dapts
from augmented.greedy import greedy_myopic_expected_utility


# -------------------------------------------------------------------
# U_PI — perfect-information (hindsight) upper bound
# -------------------------------------------------------------------

def test_u_pi_hand_computed_deterministic():
    # n=2, B=1, G=1: one test, pool of one. PI knows Z and clears the best
    # clean person. p=[0.5, 0.5], u=[3, 1]:
    #   Z={} (w .25): clears u=3 -> 3;  Z={0} (.25): clears u=1 -> 1
    #   Z={1} (.25): clears u=3 -> 3;  Z={0,1} (.25): nothing clean -> 0
    # U_PI = .25*(3+1+3+0) = 1.75
    from augmented.certificates import u_pi_exact
    val = u_pi_exact([0.5, 0.5], [3.0, 1.0], B=1, G=1)
    assert abs(val - 1.75) < 1e-12, val


def test_u_pi_caps_at_budget_times_poolsize():
    # All clean with certainty: PI clears the top B*G utilities.
    from augmented.certificates import u_pi_exact
    val = u_pi_exact([0.0, 0.0, 0.0, 0.0], [5.0, 4.0, 3.0, 2.0], B=1, G=2)
    assert abs(val - 9.0) < 1e-12, val   # top 2 = 5+4


def test_u_pi_dominates_optimum_random_instances():
    # Validity: U_PI >= U_DA on every instance (the PI adversary can replay
    # any adapted policy).
    from augmented.certificates import u_pi_exact
    rng = random.Random(2026)
    for k in range(12):
        n = rng.choice([3, 4, 5])
        B = rng.choice([1, 2])
        G = rng.choice([2, 3])
        p = [rng.uniform(0.05, 0.95) for _ in range(n)]
        u = [rng.uniform(0.5, 5.0) for _ in range(n)]
        opt, _ = solve_optimal_dapts(p, u, B, G)
        upi = u_pi_exact(p, u, B, G)
        assert upi >= opt - 1e-9, (
            f"instance {k}: U_PI={upi:.6f} < OPT={opt:.6f} "
            f"(n={n},B={B},G={G},p={p},u={u})")


def test_u_pi_mc_matches_exact():
    from augmented.certificates import u_pi_exact, u_pi_mc
    p = [0.3, 0.6, 0.2, 0.5]
    u = [1.0, 4.0, 2.0, 3.0]
    exact = u_pi_exact(p, u, B=2, G=2)
    mc = u_pi_mc(p, u, B=2, G=2, num_samples=200000, seed=0)
    assert abs(mc - exact) < 0.02, (mc, exact)


# -------------------------------------------------------------------
# U_pen — penalized information-relaxation bound (Brown–Smith–Sun style)
# -------------------------------------------------------------------

def test_u_pen_zero_penalty_equals_u_pi():
    # With V-hat identically zero the penalty vanishes and the inner problem
    # reduces to pure hindsight, whose optimum is clearing the top clean
    # utilities — so U_pen must coincide with U_PI.
    from augmented.certificates import u_pi_exact, u_pen_exact
    p = [0.4, 0.3, 0.6]
    u = [2.0, 1.0, 3.0]
    upi = u_pi_exact(p, u, B=2, G=2)
    upen = u_pen_exact(p, u, B=2, G=2, v_hat="zero")
    assert abs(upen - upi) < 1e-9, (upen, upi)


def test_u_pen_dominates_optimum_random_instances():
    # THE validity property: any martingale-difference penalty keeps
    # U_pen >= OPT. If this fails the penalty implementation is wrong.
    from augmented.certificates import u_pen_exact
    rng = random.Random(7)
    for k in range(8):
        n = rng.choice([3, 4])
        B = rng.choice([1, 2])
        G = rng.choice([2, 3])
        p = [rng.uniform(0.05, 0.95) for _ in range(n)]
        u = [rng.uniform(0.5, 5.0) for _ in range(n)]
        opt, _ = solve_optimal_dapts(p, u, B, G)
        upen = u_pen_exact(p, u, B, G, v_hat="umax")
        assert upen >= opt - 1e-9, (
            f"instance {k}: U_pen={upen:.6f} < OPT={opt:.6f} "
            f"(n={n},B={B},G={G},p={p},u={u}) — invalid penalty")


def test_u_pen_tightens_on_reference_instance():
    # The whole point: on at least a canonical instance the penalized bound
    # must be strictly tighter than raw hindsight.
    from augmented.certificates import u_pi_exact, u_pen_exact
    p = [0.3, 0.4, 0.5, 0.25, 0.6]
    u = [3.0, 1.0, 2.0, 4.0, 2.5]
    upi = u_pi_exact(p, u, B=2, G=3)
    upen = u_pen_exact(p, u, B=2, G=3, v_hat="umax")
    assert upen < upi - 1e-6, (
        f"U_pen={upen:.6f} does not tighten U_PI={upi:.6f}")


def test_u_pen_greedy_vhat_dominates_optimum():
    # Validity must hold for ANY V-hat, including the greedy value-to-go.
    from augmented.certificates import u_pen_exact
    rng = random.Random(11)
    for k in range(6):
        n = rng.choice([3, 4])
        B = rng.choice([1, 2])
        G = rng.choice([2, 3])
        p = [rng.uniform(0.05, 0.95) for _ in range(n)]
        u = [rng.uniform(0.5, 5.0) for _ in range(n)]
        opt, _ = solve_optimal_dapts(p, u, B, G)
        upen = u_pen_exact(p, u, B, G, v_hat="greedy")
        assert upen >= opt - 1e-9, (
            f"instance {k}: U_pen(greedy)={upen:.6f} < OPT={opt:.6f} "
            f"(n={n},B={B},G={G},p={p},u={u}) — invalid penalty")


def test_u_pen_vhat_comparison_documented():
    # Empirical finding (2026-07-07): the simple posterior potential ("umax",
    # linear in the exact marginals) certifies TIGHTER than the greedy
    # value-to-go ("greedy") on the reference instance (7.51 vs 8.43). The
    # greedy V-hat feeds posterior marginals into the greedy as if they were
    # independent priors; the inner adversary exploits that independence error
    # for negative penalties. Both remain valid bounds — this test documents
    # the ordering so a silent change is noticed, not because it is desirable.
    from augmented.certificates import u_pen_exact
    p = [0.3, 0.4, 0.5, 0.25, 0.6]
    u = [3.0, 1.0, 2.0, 4.0, 2.5]
    pen_umax = u_pen_exact(p, u, B=2, G=3, v_hat="umax")
    pen_greedy = u_pen_exact(p, u, B=2, G=3, v_hat="greedy")
    assert pen_umax <= pen_greedy + 1e-6, (
        f"ordering flipped: umax={pen_umax:.4f} greedy={pen_greedy:.4f} — "
        "update the finding note if the greedy V-hat now certifies tighter")


def test_vhat_registry_custom_function_still_yields_valid_bound():
    # La frontera de seguridad del harness de autoresearch: CUALQUIER V-hat
    # registrada — incluso una arbitraria y mal calibrada — debe producir una
    # cota valida, porque la penalizacion es una diferencia de martingala por
    # construccion. Solo la tightness varia con V-hat, nunca la validez.
    from augmented.vhat import register, VHAT_REGISTRY
    from augmented.certificates import u_pen_exact

    name = "_test_weird_vhat"
    if name not in VHAT_REGISTRY:
        @register(name)
        def _weird(ctx, h_fs, remaining):
            # deliberadamente rara: escala grande, mezcla tamano de historia
            # con utilidades, ignora el posterior
            return 7.3 * len(h_fs) + 0.5 * sum(ctx.u) * (remaining + 1)

    rng = random.Random(23)
    for k in range(5):
        n = rng.choice([3, 4])
        B = rng.choice([1, 2])
        G = rng.choice([2, 3])
        p = [rng.uniform(0.05, 0.95) for _ in range(n)]
        u = [rng.uniform(0.5, 5.0) for _ in range(n)]
        opt, _ = solve_optimal_dapts(p, u, B, G)
        upen = u_pen_exact(p, u, B, G, v_hat=name)
        assert upen >= opt - 1e-9, (
            f"instance {k}: U_pen(custom)={upen:.6f} < OPT={opt:.6f} — "
            "una V-hat registrada rompio la validez; la frontera del teorema "
            "esta mal implementada")


def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
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
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

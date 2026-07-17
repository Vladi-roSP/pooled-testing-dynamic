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


# -------------------------------------------------------------------
# Task 4: posterior_draws + selector por frecuencia conjunta
# -------------------------------------------------------------------

def test_posterior_draws_match_exact_marginals():
    from augmented.bayesian import posterior_draws, bayesian_update_by_counting
    rng = random.Random(11)
    n = 8
    p = [rng.uniform(0.1, 0.5) for _ in range(n)]
    history = ((0b00000111, 1), (0b00011100, 1))  # overlapping, correlated
    exact = bayesian_update_by_counting(p, history, n)
    draws = posterior_draws(p, history, n, num_draws=20000, seed=3)
    assert len(draws) == 20000
    for i in range(n):
        freq = sum(1 for z in draws if z >> i & 1) / len(draws)
        assert abs(freq - exact[i]) < 0.02, (i, freq, exact[i])


def test_posterior_draws_joint_frequency_beats_product_on_known_case():
    # history {0,1}=1, {1,2}=1 with p=0.15: worlds are (0,1,0) w/ 0.85 and
    # (1,0,1) w/ 0.15. P(0 and 2 both clean) = 0.85; the product says 0.7225.
    from augmented.bayesian import posterior_draws
    p = [0.15, 0.15, 0.15]
    history = ((0b011, 1), (0b110, 1))
    draws = posterior_draws(p, history, 3, num_draws=20000, seed=5)
    joint = sum(1 for z in draws if (z & 0b101) == 0) / len(draws)
    assert abs(joint - 0.85) < 0.02


def test_sample_best_pool_picks_true_argmax_on_small_case():
    from augmented.bayesian import posterior_draws
    from augmented.pool_solvers import sample_best_pool
    p = [0.15, 0.15, 0.15]
    u = [1.0, 1.0, 1.0]
    history = ((0b011, 1), (0b110, 1))
    draws = posterior_draws(p, history, 3, num_draws=5000, seed=8)
    pool = sample_best_pool(draws, u, G=2, n=3, cleared_mask=0)
    # True best pool is {0,2}: P(clean)=0.85, gain 2 -> score 1.70;
    # any pool containing 1 has P(clean)<=0.15.
    assert pool == 0b101, bin(pool)


# -------------------------------------------------------------------
# Task 6: pesos de rama exactos en hybrid_solver (invariante K=B)
# -------------------------------------------------------------------

def test_hybrid_kB_equals_greedy_on_overlap_heavy_instances():
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    from augmented.greedy import greedy_myopic_expected_utility
    for seed in range(10):
        rng = random.Random(400 + seed)
        n = 5 if seed % 2 else 6
        p = [rng.uniform(0.2, 0.7) for _ in range(n)]  # high p -> overlap-rich
        u = [rng.uniform(1.0, 5.0) for _ in range(n)]
        g = greedy_myopic_expected_utility(p, u, 3, 3)
        _, h = hybrid_greedy_bruteforce(p, u, 3, 3, greedy_steps=3)  # K=B
        assert abs(g - h) < 1e-9, (seed, g, h)


# -------------------------------------------------------------------
# Task 7: fase DP condicionada en la historia real
# -------------------------------------------------------------------

def test_hybrid_k0_equals_exact_dp():
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    from augmented.solver import solve_optimal_dapts
    for seed in range(5):
        rng = random.Random(500 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        opt, _ = solve_optimal_dapts(p, u, 3, 3)
        _, h = hybrid_greedy_bruteforce(p, u, 3, 3, greedy_steps=0)
        assert abs(opt - h) < 1e-9, (seed, opt, h)


def test_hybrid_midK_between_greedy_and_opt():
    # greedy prefix + optimal suffix must sit between full greedy and OPT
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    from augmented.greedy import greedy_myopic_expected_utility
    from augmented.solver import solve_optimal_dapts
    for seed in range(5):
        rng = random.Random(500 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        g = greedy_myopic_expected_utility(p, u, 3, 3)
        opt, _ = solve_optimal_dapts(p, u, 3, 3)
        for K in (1, 2):
            _, h = hybrid_greedy_bruteforce(p, u, 3, 3, greedy_steps=K)
            assert g - 1e-9 <= h <= opt + 1e-9, (seed, K, g, h, opt)


def _eval_policy_tree(tree, p, u, n):
    """Exact E[utility] of the deterministic policy encoded by a tree_dict:
    walk every latent profile z through the tree, credit r=0 pools."""
    from augmented.core import popcount
    total = 0.0
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else (1.0 - p[i])
        if w == 0.0:
            continue
        node, cleared = tree, 0
        while not node['terminal']:
            pool = node['pool']
            r = popcount(pool & z)
            if r == 0:
                cleared |= pool
            nxt = node['children'].get(r)
            if nxt is None:
                break  # branch missing for a reachable outcome: value is lost
            node = nxt
        total += w * sum(u[i] for i in range(n) if cleared >> i & 1)
    return total


def test_hybrid_reported_value_matches_true_policy_value():
    # The DP phase must be conditioned on the real history: the reported EU
    # has to equal the exact value of the policy tree it returns.
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    for seed in range(6):
        rng = random.Random(520 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        for K in (1, 2):
            tree, h = hybrid_greedy_bruteforce(p, u, 3, 3, greedy_steps=K)
            true_val = _eval_policy_tree(tree, p, u, 5)
            assert abs(h - true_val) < 1e-9, (seed, K, h, true_val)


# -------------------------------------------------------------------
# Task 8: pesos de rama exactos en semi_utility
# -------------------------------------------------------------------

def _bruteforce_semi(p, u, B, G, alpha, mode):
    """Exact E[utility] of the semi-utility greedy: enumerate every latent
    profile, run the very same policy via simulate, weight by the prior."""
    from augmented.semi_utility import greedy_myopic_semi_simulate
    n = len(p)
    total = 0.0
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else (1.0 - p[i])
        if w == 0.0:
            continue
        _, _, val = greedy_myopic_semi_simulate(p, u, B, G, z, alpha,
                                                update_method=mode)
        total += w * val
    return total


def test_semi_utility_matches_bruteforce_all_modes():
    from augmented.semi_utility import greedy_myopic_semi_expected_utility
    for seed in range(8):
        rng = random.Random(600 + seed)
        n = 5 if seed % 2 else 6
        p = [rng.uniform(0.2, 0.7) for _ in range(n)]
        u = [rng.uniform(1.0, 5.0) for _ in range(n)]
        for alpha in (0.5, 1.0):
            for mode in ("sequential", "counting"):
                eu = greedy_myopic_semi_expected_utility(p, u, 3, 3,
                                                         alpha=alpha,
                                                         update_method=mode)
                bf = _bruteforce_semi(p, u, 3, 3, alpha, mode)
                assert abs(eu - bf) < 1e-9, (seed, alpha, mode, eu, bf)


def test_semi_utility_counting_mode_no_crash():
    from augmented.semi_utility import greedy_myopic_semi_expected_utility
    for seed in range(20):
        rng = random.Random(700 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        greedy_myopic_semi_expected_utility(p, u, 3, 3, alpha=0.3,
                                            update_method="counting")  # must not raise


# -------------------------------------------------------------------
# Task 9: state_reward_greedy — rama exacta hasta n=18, MC honesto arriba
# -------------------------------------------------------------------

def _bruteforce_beta(p, u, B, G, beta, info_metric):
    """Exact E[utility] of the beta-reward greedy via profile enumeration."""
    from augmented.state_reward_greedy import greedy_myopic_beta_simulate
    n = len(p)
    total = 0.0
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else (1.0 - p[i])
        if w == 0.0:
            continue
        _, _, val = greedy_myopic_beta_simulate(p, u, B, G, z, beta,
                                                info_metric)
        total += w * val
    return total


def test_beta_eu_exact_matches_bruteforce_small_n():
    from augmented.state_reward_greedy import greedy_myopic_beta_expected_utility
    from augmented.greedy import greedy_myopic_expected_utility
    for seed in range(6):
        rng = random.Random(750 + seed)
        n = 5 if seed % 2 else 6
        p = [rng.uniform(0.2, 0.7) for _ in range(n)]  # overlap-heavy
        u = [rng.uniform(1.0, 5.0) for _ in range(n)]
        for beta in (0.0, 0.5):
            eu = greedy_myopic_beta_expected_utility(p, u, 3, 3, beta=beta)
            bf = _bruteforce_beta(p, u, 3, 3, beta, 'entropy')
            assert abs(eu - bf) < 1e-9, (seed, beta, eu, bf)
        # beta=0 must ALSO equal the standard greedy EU (legacy invariant,
        # now to 1e-9 on overlap-heavy instances)
        g = greedy_myopic_expected_utility(p, u, 3, 3)
        eu0 = greedy_myopic_beta_expected_utility(p, u, 3, 3, beta=0.0)
        assert abs(g - eu0) < 1e-9, (seed, g, eu0)


def test_beta_eu_large_n_reports_se_and_uses_enough_trials():
    from augmented.state_reward_greedy import greedy_myopic_beta_expected_utility
    rng = random.Random(910)
    n = 20
    p = [rng.uniform(0.05, 0.4) for _ in range(n)]
    u = [rng.uniform(1.0, 5.0) for _ in range(n)]
    m1, se1 = greedy_myopic_beta_expected_utility(p, u, 4, 3, beta=0.5,
                                                  return_se=True, seed=11)
    m2, se2 = greedy_myopic_beta_expected_utility(p, u, 4, 3, beta=0.5,
                                                  return_se=True, seed=11)
    assert (m1, se1) == (m2, se2)          # seeded reproducibility
    assert 0.0 < se1 < 0.05 * m1           # honest SE, tight with 200 trials
    m3 = greedy_myopic_beta_expected_utility(p, u, 4, 3, beta=0.5, seed=11)
    assert m3 == m1                        # legacy scalar return by default


# -------------------------------------------------------------------
# Task 10: gates de los runners alineados a la frontera exacta
# -------------------------------------------------------------------

def test_experiment_gates_match_exact_pmf_frontier():
    from augmented.greedy import EXACT_PMF_MAX_N
    import augmented.sprint3_experiments as s3
    import augmented.overnight_experiments as ov
    assert EXACT_PMF_MAX_N == 18
    assert s3.exact_eu_feasible(18) and not s3.exact_eu_feasible(19)
    assert ov.exact_eu_feasible(18) and not ov.exact_eu_feasible(19)

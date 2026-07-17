"""
Greedy adaptive group counting with an information-reward meta-parameter beta.

Run with: python augmented/state_reward_greedy.py
"""

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (all_pools_from_mask, compute_active_mask,
                            indices_from_mask, mask_str, test_result)
from augmented.bayesian import bayesian_update_single_test, _poisson_binomial_pmf
from augmented.baselines import u_max
from augmented.greedy import EXACT_PMF_MAX_N, _branch_pmf


_EXACT_POOL_LIMIT = 14
_HEURISTIC_SHORTLIST = 8
_EXACT_POOL_ENUM_LIMIT = 50000


def _safe_binary_entropy(x):
    """Binary entropy with guards for x=0 or x=1."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log(x + 1e-15) + (1.0 - x) * math.log(1.0 - x + 1e-15))


def _uncertainty_proxy(x, info_metric):
    if info_metric == 'entropy':
        return _safe_binary_entropy(x)
    if info_metric == 'variance':
        return x * (1.0 - x)
    if info_metric == 'confirmed':
        return x * (1.0 - x)
    raise ValueError(f"Unknown info_metric: {info_metric}")


def _candidate_pools(p, u, G, n, cleared_mask, beta, info_metric):
    """Enumerate exact pools for small states and shortlist for large ones."""
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    if active_mask == 0:
        return active_mask, []

    active_idx = indices_from_mask(active_mask, n)
    total_pools = sum(math.comb(len(active_idx), k)
                      for k in range(1, min(G, len(active_idx)) + 1))
    if len(active_idx) <= _EXACT_POOL_LIMIT or total_pools <= _EXACT_POOL_ENUM_LIMIT:
        return active_mask, all_pools_from_mask(active_mask, G, include_empty=False)

    shortlist_size = min(len(active_idx), _HEURISTIC_SHORTLIST)
    ranked = sorted(
        active_idx,
        key=lambda i: ((1.0 - p[i]) * u[i] + beta * _uncertainty_proxy(p[i], info_metric)),
        reverse=True,
    )[:shortlist_size]

    pools = []
    for k in range(1, min(G, len(ranked)) + 1):
        for combo in itertools.combinations(ranked, k):
            pool = 0
            for i in combo:
                pool |= 1 << i
            pools.append(pool)
    return active_mask, pools


def _draw_z_mask_from_prior(rng, p):
    z_mask = 0
    for i, pi in enumerate(p):
        if rng.random() < pi:
            z_mask |= 1 << i
    return z_mask


def _compute_info_gain(pool, p, n, info_metric, cleared_mask):
    """Compute expected information gain for a candidate pool."""
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    pool_idx = indices_from_mask(pool, n)
    if not pool_idx:
        return 0.0

    active_idx = indices_from_mask(active_mask, n)
    pmf = _poisson_binomial_pmf([p[i] for i in pool_idx])

    if info_metric == 'confirmed':
        before_confirmed = {
            i for i in active_idx if p[i] > 0.95 or p[i] < 0.05
        }
        expected_gain = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            post = bayesian_update_single_test(p, pool, r, n)
            after_confirmed = {
                i for i in active_idx if post[i] > 0.95 or post[i] < 0.05
            }
            expected_gain += pmf[r] * len(after_confirmed - before_confirmed)
        return expected_gain

    if info_metric == 'entropy':
        before = sum(_safe_binary_entropy(p[i]) for i in active_idx)
        expected_after = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            post = bayesian_update_single_test(p, pool, r, n)
            expected_after += pmf[r] * sum(
                _safe_binary_entropy(post[i]) for i in active_idx)
        return before - expected_after

    if info_metric == 'variance':
        before = sum(p[i] * (1.0 - p[i]) for i in active_idx)
        expected_after = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            post = bayesian_update_single_test(p, pool, r, n)
            expected_after += pmf[r] * sum(
                post[i] * (1.0 - post[i]) for i in active_idx)
        return before - expected_after

    raise ValueError(f"Unknown info_metric: {info_metric}")


def _beta_best_pool(p, u, G, n, cleared_mask, beta, info_metric):
    """Pick pool maximizing immediate clear reward plus beta information gain."""
    active_mask, pools = _candidate_pools(p, u, G, n, cleared_mask, beta, info_metric)
    if active_mask == 0:
        return 0
    best_pool, best_score = 0, 0.0

    for pool in pools:
        pool_idx = indices_from_mask(pool, n)
        prob_clear = 1.0
        for i in pool_idx:
            prob_clear *= (1.0 - p[i])

        gain = sum(u[i] for i in pool_idx if not (cleared_mask >> i & 1))
        info_gain = _compute_info_gain(pool, p, n, info_metric, cleared_mask)
        score = prob_clear * gain + beta * info_gain

        if score > best_score:
            best_score = score
            best_pool = pool

    return best_pool


def greedy_myopic_beta_simulate(p, u, B, G, z_mask, beta, info_metric='entropy'):
    """Simulate myopic beta-reward greedy on a fixed latent-state profile."""
    n = len(p)
    current_p = list(p)
    cleared_mask = 0
    history = ()

    for _ in range(B):
        pool = _beta_best_pool(current_p, u, G, n, cleared_mask, beta, info_metric)
        if pool == 0:
            break
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool
        current_p = bayesian_update_single_test(current_p, pool, r, n)

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


def greedy_myopic_beta_expected_utility(p, u, B, G, beta, info_metric='entropy',
                                        num_trials=200, seed=0,
                                        return_se=False):
    """Expected utility of myopic beta-reward greedy.

    Exact (branch pmf sobre perfiles consistentes) hasta
    EXACT_PMF_MAX_N agentes activos; arriba de eso, media Monte Carlo con
    ``num_trials`` simulaciones sembradas con ``seed``.

    return_se=True devuelve (media, error estandar); el SE es 0.0 en la rama
    exacta. Por omision devuelve solo la media (compatibilidad hacia atras).
    """
    n = len(p)

    active_mask, _ = compute_active_mask(p, 0, n)
    if len(indices_from_mask(active_mask, n)) > EXACT_PMF_MAX_N:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(num_trials):
            z_mask = _draw_z_mask_from_prior(rng, p)
            _, _, utility = greedy_myopic_beta_simulate(
                p, u, B, G, z_mask, beta, info_metric)
            vals.append(utility)
        mean = sum(vals) / num_trials
        var = (sum((v - mean) ** 2 for v in vals) / (num_trials - 1)
               if num_trials > 1 else 0.0)
        se = (var / num_trials) ** 0.5
        return (mean, se) if return_se else mean

    prior = list(p)

    def recurse(current_p, history, b, cleared_mask):
        if b == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        pool = _beta_best_pool(current_p, u, G, n, cleared_mask, beta, info_metric)
        if pool == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        pool_idx = indices_from_mask(pool, n)
        pmf = _branch_pmf(prior, history, current_p, pool, pool_idx, n)

        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(current_p, pool, r, n)
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_history = history + ((pool, r),)
            ev += pmf[r] * recurse(new_p, new_history, b - 1, new_cleared)
        return ev

    val = recurse(list(p), (), B, 0)
    return (val, 0.0) if return_se else val


def _vip_scenario():
    p_vip = [0.8] * 8
    u_vip = [10.0] * 8
    p_reg = [0.2] * 12
    u_reg = [2.0] * 12
    p = p_vip + p_reg
    u = u_vip + u_reg
    return p, u, 6, 10


def _vip_moderate_scenario():
    p_vip = [0.35] * 8
    u_vip = [10.0] * 8
    p_reg = [0.1] * 12
    u_reg = [2.0] * 12
    p = p_vip + p_reg
    u = u_vip + u_reg
    return p, u, 6, 5


def run_vip_benchmark(beta_values=None, info_metric='confirmed', seed=42):
    """Run Francisco's VIP benchmark and print a comparison table."""
    if beta_values is None:
        beta_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    np.random.seed(seed)
    scenarios = [
        ("VIP-high-prev", _vip_scenario()),
        ("VIP-moderate-prev", _vip_moderate_scenario()),
    ]

    print("VIP benchmark")
    for scenario_name, (p, u, B, G) in scenarios:
        n = len(p)
        print(scenario_name)
        print(f"u_max={u_max(p, u):.6f}")
        if scenario_name == "VIP-high-prev":
            print(
                "diagnostic: high VIP prevalence makes P(r=0) shrink exponentially "
                "with pool size, so the immediate myopic term dominates beta."
            )
        print("beta | expected_utility | first_pool | first_pool_size")
        for beta in beta_values:
            eu = greedy_myopic_beta_expected_utility(p, u, B, G, beta, info_metric)
            first_pool = _beta_best_pool(p, u, G, n, 0, beta, info_metric)
            first_pool_size = len(indices_from_mask(first_pool, n))
            print(
                f"{beta:4.1f} | {eu:16.6f} | {mask_str(first_pool, n):10s} | "
                f"{first_pool_size}"
            )
        print()


def _draw_vip_instance(seed):
    rng = np.random.default_rng(seed)
    p_vip = np.clip(0.8 + rng.uniform(-0.05, 0.05, size=8), 0.05, 0.95).tolist()
    p_reg = np.clip(0.2 + rng.uniform(-0.05, 0.05, size=12), 0.05, 0.95).tolist()
    u_vip = rng.uniform(8.0, 12.0, size=8).tolist()
    u_reg = rng.uniform(1.0, 3.0, size=12).tolist()
    return p_vip + p_reg, u_vip + u_reg, 6, 10


def _draw_uniform_instance(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.4, size=15).tolist()
    u = [1.0] * 15
    return p, u, 5, 5


def run_beta_sweep(n_instances=30, seed=0):
    """Run a randomized beta sweep for VIP and uniform scenarios."""
    beta_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    scenarios = [
        ("VIP", _draw_vip_instance),
        ("uniform", _draw_uniform_instance),
    ]

    print("beta sweep")
    print("scenario | beta | mean_EU | std_EU")
    for scenario_name, scenario_fn in scenarios:
        for beta in beta_values:
            eus = []
            for instance_idx in range(n_instances):
                instance_seed = seed + instance_idx
                p, u, B, G = scenario_fn(instance_seed)
                eu = greedy_myopic_beta_expected_utility(
                    p, u, B, G, beta, info_metric='entropy')
                eus.append(eu)
            print(
                f"{scenario_name:8s} | {beta:4.1f} | "
                f"{float(np.mean(eus)):7.4f} +/- {float(np.std(eus)):.4f}"
            )
    print()


if __name__ == '__main__':
    run_vip_benchmark()
    run_beta_sweep(n_instances=2)
